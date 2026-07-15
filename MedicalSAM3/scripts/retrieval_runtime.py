"""检索条件 MedEx-SAM3 脚本的共享运行时辅助工具。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from MedicalSAM3.adapters import RetrievalSpatialSemanticAdapter
from MedicalSAM3.adapters.lora import LoRAConfig, apply_lora_to_model, load_lora_weights
from MedicalSAM3.exemplar_bank import RSSDABank
from MedicalSAM3.models.retrieval import PrototypeRetriever, SimilarityHeatmapBuilder
from MedicalSAM3.retrieval import (
    DirectoryBankLoader,
    LoadedBankContext,
    SiteBankResolution,
    annotate_single_bank_retrieval,
    fuse_multi_bank_retrieval,
    load_retrieval_bank,
    resolve_site_bank_paths,
)
from MedicalSAM3.retrieval.mask_prior import attach_retrieved_mask_priors
from MedicalSAM3.sam3_official.build_model import build_official_sam3_image_model, freeze_model
from MedicalSAM3.sam3_official.tensor_forward import Sam3TensorForwardWrapper
from MedicalSAM3.scripts.common import resolve_feature_map

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass
class RetrievalBankBackend:
    """封装单个检索库后端的上下文、检索器与加载器。

    参数：
        - 无

    返回：
        - 提供用于检索流程的运行时后端实例
    """
    name: str
    bank_context: LoadedBankContext
    retriever: PrototypeRetriever
    retrieval_backend: str
    directory_loader: Optional[DirectoryBankLoader] = None


@dataclass
class RetrievalRuntime:
    """检索运行时数据类，聚合模型、检索器、适配器及库上下文。

    参数：
        - 无

    返回：
        - 提供用于推理流程的完整运行时实例
    """
    device: str
    hidden_dim: int
    bank_context: LoadedBankContext
    wrapper: Sam3TensorForwardWrapper
    retriever: PrototypeRetriever
    similarity_builder: SimilarityHeatmapBuilder
    adapter: RetrievalSpatialSemanticAdapter
    retrieval_backend: str
    directory_loader: Optional[DirectoryBankLoader] = None
    primary_backend: Optional[RetrievalBankBackend] = None
    site_bank_mode: str = "train_plus_site"
    continual_bank_root: Optional[Path] = None
    bank_loader_config: dict[str, Any] = field(default_factory=dict)
    bank_backend_cache: dict[str, RetrievalBankBackend] = field(default_factory=dict)


def resolve_hidden_dim(model: torch.nn.Module) -> int:
    """从模型中解析隐藏层维度。

    参数：
        - model: PyTorch 模型

    返回：
        - 隐藏层维度整数值
    """
    return int(getattr(model, "hidden_dim", getattr(model, "_medex_hidden_dim", getattr(model, "embed_dim", 128))))


def apply_retrieval_mode(retrieval: dict[str, Any], mode: str) -> dict[str, Any]:
    """根据检索模式过滤检索结果（positive-only 清空负例，negative-only 清空正例）。

    参数：
        - retrieval: 原始检索结果
        - mode: 检索模式

    返回：
        - 应用模式后的检索结果
    """
    if mode in {"joint", "semantic", "spatial", "positive-negative"}:
        return retrieval
    if mode not in {"positive-only", "negative-only"}:
        raise ValueError(f"Unsupported retrieval mode: {mode}")
    updated = dict(retrieval)
    if mode == "positive-only":
        updated["negative_features"] = torch.zeros_like(retrieval["negative_features"])
        updated["negative_weights"] = torch.zeros_like(retrieval["negative_weights"])
        updated["negative_score_tensor"] = torch.zeros_like(retrieval.get("negative_score_tensor", retrieval["negative_weights"]))
        updated["negative_prototype"] = torch.zeros_like(retrieval["positive_prototype"])
        updated["negative_entries"] = [[] for _ in retrieval["positive_entries"]]
        updated["negative_scores"] = [torch.zeros_like(score) for score in retrieval["positive_scores"]]
        if "negative_mask_prior" in retrieval:
            updated["negative_mask_prior"] = torch.zeros_like(retrieval["negative_mask_prior"])
        return updated
    updated["positive_features"] = torch.zeros_like(retrieval["positive_features"])
    updated["positive_weights"] = torch.zeros_like(retrieval["positive_weights"])
    updated["positive_score_tensor"] = torch.zeros_like(retrieval.get("positive_score_tensor", retrieval["positive_weights"]))
    updated["positive_prototype"] = torch.zeros_like(retrieval["negative_prototype"])
    updated["positive_entries"] = [[] for _ in retrieval["negative_entries"]]
    updated["positive_scores"] = [torch.zeros_like(score) for score in retrieval["negative_scores"]]
    if "positive_mask_prior" in retrieval:
        updated["positive_mask_prior"] = torch.zeros_like(retrieval["positive_mask_prior"])
    return updated


def parse_bbox(value: str) -> list[float]:
    """将逗号分隔的边界框字符串解析为浮点数列表。

    参数：
        - value: "x1,y1,x2,y2" 格式的字符串

    返回：
        - [x1, y1, x2, y2] 浮点数列表
    """
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != 4:
        raise ValueError("bbox must be formatted as x1,y1,x2,y2")
    return [float(part) for part in parts]


def load_bbox_mapping(path: str | Path) -> dict[str, list[float]]:
    """从 JSON 文件加载图像到边界框的映射。

    参数：
        - path: JSON 文件路径

    返回：
        - 图像标识符到 [x1,y1,x2,y2] 的字典
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return {str(key): [float(item) for item in value] for key, value in payload.items()}
    if isinstance(payload, list):
        mapping: dict[str, list[float]] = {}
        for item in payload:
            if not isinstance(item, dict):
                continue
            key = str(item.get("image") or item.get("image_id") or item.get("path") or "")
            bbox = item.get("bbox")
            if key and isinstance(bbox, list) and len(bbox) == 4:
                mapping[key] = [float(entry) for entry in bbox]
        return mapping
    raise ValueError(f"Unsupported bbox mapping payload: {path}")


def collect_input_images(input_path: str | Path) -> list[Path]:
    """收集输入路径下的所有支持格式的图像文件。

    参数：
        - input_path: 输入文件或目录路径

    返回：
        - 图像文件路径列表
    """
    target = Path(input_path)
    if target.is_file():
        return [target]
    if not target.is_dir():
        raise FileNotFoundError(f"Input path not found: {target}")
    return sorted(path for path in target.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES)


def load_image_tensor(image_path: str | Path, image_size: int) -> tuple[Image.Image, torch.Tensor]:
    """从文件加载图像并缩放到指定尺寸，返回 PIL 图像和归一化张量。

    参数：
        - image_path: 图像文件路径
        - image_size: 目标尺寸

    返回：
        - (PIL Image, 归一化张量 [1,3,H,W]) 的元组
    """
    image = Image.open(image_path).convert("RGB")
    resized = image.resize((image_size, image_size))
    array = np.asarray(resized).astype("float32") / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return image, tensor


def scale_bbox(bbox: list[float], original_size: tuple[int, int], image_size: int) -> torch.Tensor:
    """将边界框从原始图像坐标缩放到目标尺寸坐标。

    参数：
        - bbox: [x1,y1,x2,y2] 原始坐标
        - original_size: 原始图像尺寸 (width, height)
        - image_size: 目标尺寸

    返回：
        - 缩放后的边界框张量
    """
    width, height = original_size
    if width <= 0 or height <= 0:
        raise ValueError("original image size must be positive")
    scale_x = image_size / float(width)
    scale_y = image_size / float(height)
    x1, y1, x2, y2 = bbox
    return torch.tensor([x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y], dtype=torch.float32)


def load_rssda_bundle_components(
    path: str | Path,
    *,
    device: str,
    adapter: RetrievalSpatialSemanticAdapter,
    retriever: PrototypeRetriever,
    similarity_builder: SimilarityHeatmapBuilder,
) -> dict[str, bool]:
    """从捆绑检查点加载 RSS-DA 各组件的状态字典。

    参数：
        - path: 检查点路径
        - device: 加载目标设备
        - adapter: RSS-DA 适配器
        - retriever: 原型检索器
        - similarity_builder: 相似度热图构建器

    返回：
        - 各组件是否加载成功的字典
    """
    payload = torch.load(path, map_location=device, weights_only=False)
    loaded = {"adapter": False, "retriever": False, "similarity_builder": False}
    if not isinstance(payload, dict):
        return loaded
    if isinstance(payload.get("adapter"), dict):
        adapter.load_state_dict(payload["adapter"], strict=False)
        loaded["adapter"] = True
    if isinstance(payload.get("retriever"), dict):
        retriever.load_state_dict(payload["retriever"], strict=False)
        loaded["retriever"] = True
    if isinstance(payload.get("similarity_builder"), dict):
        similarity_builder.load_state_dict(payload["similarity_builder"], strict=False)
        loaded["similarity_builder"] = True
    return loaded


def _build_bank_backend(
    *,
    name: str,
    bank_context: LoadedBankContext,
    feature_dim: int,
    top_k_positive: int,
    top_k_negative: int,
    device: str,
    checkpoint: Optional[str],
    precision: str,
    image_size: int,
    allow_dummy_fallback: bool,
    reference_retriever: PrototypeRetriever,
    fallback_backend: str,
) -> RetrievalBankBackend:
    """构建单个检索库后端，包含检索器及可选的目录加载器。

    参数：
        - name: 后端名称
        - bank_context: 已加载的库上下文
        - feature_dim: 特征维度
        - top_k_positive: 选取正例数量
        - top_k_negative: 选取负例数量
        - device: 设备
        - checkpoint: 检查点路径
        - precision: 精度设置
        - image_size: 图像尺寸
        - allow_dummy_fallback: 是否允许虚拟回退
        - reference_retriever: 参考检索器（用于复制权重）
        - fallback_backend: 回退后端类型

    返回：
        - RetrievalBankBackend 实例
    """
    retriever = PrototypeRetriever(
        bank=bank_context.bank,
        feature_dim=feature_dim,
        top_k_positive=top_k_positive,
        top_k_negative=top_k_negative,
    ).to(device)
    retriever.load_state_dict(reference_retriever.state_dict(), strict=False)
    retriever.eval()
    directory_loader = None
    retrieval_backend = fallback_backend
    if bank_context.source == "directory_bank" and fallback_backend == "directory_loader":
        directory_loader = DirectoryBankLoader(
            bank_context.resolved_path,
            checkpoint=checkpoint,
            device=device,
            precision=precision,
            image_size=image_size,
            allow_dummy_fallback=allow_dummy_fallback,
            default_top_k=max(top_k_positive, top_k_negative),
        )
        retrieval_backend = "directory_loader"
    return RetrievalBankBackend(
        name=name,
        bank_context=bank_context,
        retriever=retriever,
        retrieval_backend=retrieval_backend,
        directory_loader=directory_loader,
    )


def _backend_cache_key(path: Path) -> str:
    """生成后端缓存键（解析后的绝对路径字符串）。

    参数：
        - path: 路径

    返回：
        - 缓存键字符串
    """
    return str(path.resolve()) if path.exists() else str(path)


def _resolve_site_bank_root(memory_bank: str | Path, explicit_root: str | Path | None) -> Path | None:
    """解析站点库根目录路径。

    参数：
        - memory_bank: 记忆库路径
        - explicit_root: 显式指定的根目录

    返回：
        - 根目录路径或 None
    """
    if explicit_root is not None:
        return Path(explicit_root)
    memory_bank_path = Path(memory_bank)
    parent = memory_bank_path.parent if memory_bank_path.name == "train_bank" else memory_bank_path
    candidate = parent / "continual_bank"
    if candidate.exists():
        return candidate
    return None


def _default_site_resolution(runtime: RetrievalRuntime) -> SiteBankResolution:
    """生成默认的站点库解析结果（仅训练库模式）。

    参数：
        - runtime: 检索运行时

    返回：
        - 默认的 SiteBankResolution 实例
    """
    return SiteBankResolution(
        mode="train_only",
        site_id=None,
        train_bank_path=runtime.bank_context.resolved_path,
        continual_bank_root=runtime.continual_bank_root or runtime.bank_context.resolved_path.parent,
        site_bank_path=None,
        expected_site_bank=None,
        selected_bank_paths=[runtime.bank_context.resolved_path],
        fallback_to_train_bank=False,
        fallback_reason=None,
        warnings=[],
    )


def _resolve_bank_selection(runtime: RetrievalRuntime, sample_metadata: dict[str, Any] | None) -> SiteBankResolution:
    """解析当前样本应使用的库选择（训练库、站点库或多库融合）。

    参数：
        - runtime: 检索运行时
        - sample_metadata: 样本元数据

    返回：
        - 库解析结果 SiteBankResolution
    """
    if runtime.continual_bank_root is None:
        return _default_site_resolution(runtime)
    return resolve_site_bank_paths(
        sample_metadata=sample_metadata,
        train_bank=runtime.bank_context.resolved_path,
        continual_bank_root=runtime.continual_bank_root,
        mode=runtime.site_bank_mode,
    )


def _get_backend(runtime: RetrievalRuntime, bank_path: Path) -> RetrievalBankBackend:
    """获取指定库路径对应的后端实例（含缓存逻辑）。

    参数：
        - runtime: 检索运行时
        - bank_path: 库路径

    返回：
        - 检索库后端实例
    """
    cache_key = _backend_cache_key(bank_path)
    if runtime.primary_backend is not None and cache_key == _backend_cache_key(runtime.bank_context.resolved_path):
        return runtime.primary_backend
    cached = runtime.bank_backend_cache.get(cache_key)
    if cached is not None:
        return cached

    config = runtime.bank_loader_config
    bank_context = load_retrieval_bank(
        bank_path,
        purpose=str(config.get("bank_purpose", "external-eval")),
        checkpoint=config.get("checkpoint"),
        device=runtime.device,
        precision=str(config.get("precision", "fp32")),
        image_size=int(config.get("image_size", 128)),
        allow_dummy_fallback=bool(config.get("allow_dummy_fallback", False)),
    )
    backend = _build_bank_backend(
        name=bank_path.name,
        bank_context=bank_context,
        feature_dim=runtime.hidden_dim,
        top_k_positive=int(config.get("top_k_positive", 1)),
        top_k_negative=int(config.get("top_k_negative", 1)),
        device=runtime.device,
        checkpoint=config.get("checkpoint"),
        precision=str(config.get("precision", "fp32")),
        image_size=int(config.get("image_size", 128)),
        allow_dummy_fallback=bool(config.get("allow_dummy_fallback", False)),
        reference_retriever=runtime.retriever,
        fallback_backend=runtime.retrieval_backend,
    )
    runtime.bank_backend_cache[cache_key] = backend
    return backend


def _run_backend_retrieval(
    backend: RetrievalBankBackend,
    query_feature: torch.Tensor,
    *,
    top_k: Optional[int],
    top_k_positive: Optional[int],
    top_k_negative: Optional[int],
    query_source: Optional[str],
    prefer_cross_domain_positive: bool,
) -> dict[str, Any]:
    """在指定后端上执行检索操作。

    参数：
        - backend: 检索库后端
        - query_feature: 查询特征
        - top_k: 总检索数量
        - top_k_positive: 正例数量
        - top_k_negative: 负例数量
        - query_source: 查询来源域
        - prefer_cross_domain_positive: 是否优先选用跨域正例

    返回：
        - 检索结果字典
    """
    if backend.retrieval_backend == "directory_loader" and backend.directory_loader is not None:
        return backend.directory_loader.retrieve(
            query_feature,
            top_k=top_k,
            top_k_positive=top_k_positive,
            top_k_negative=top_k_negative,
            query_source_datasets=[query_source] if query_source else None,
            prefer_cross_domain_positive=prefer_cross_domain_positive,
        )
    return backend.retriever(
        query_feature,
        top_k_positive=top_k_positive or top_k,
        top_k_negative=top_k_negative or top_k,
        query_source_datasets=[query_source] if query_source else None,
        prefer_cross_domain_positive=prefer_cross_domain_positive,
    )


def resolve_effective_bank(runtime: RetrievalRuntime, *, sample_metadata: dict[str, Any] | None = None) -> RSSDABank:
    """解析当前样本应使用的有效 RSS-DA 库（单库或融合多库）。

    参数：
        - runtime: 检索运行时
        - sample_metadata: 样本元数据

    返回：
        - RSSDABank 实例（包含去重的条目）
    """
    resolution = _resolve_bank_selection(runtime, sample_metadata)
    if len(resolution.selected_bank_paths) <= 1:
        selected_path = resolution.selected_bank_paths[0] if resolution.selected_bank_paths else runtime.bank_context.resolved_path
        return _get_backend(runtime, selected_path).bank_context.bank
    entries = []
    seen: set[tuple[str, str]] = set()
    for bank_path in resolution.selected_bank_paths:
        backend = _get_backend(runtime, bank_path)
        for entry in backend.bank_context.bank.entries:
            key = (entry.prototype_id, entry.feature_path)
            if key in seen:
                continue
            seen.add(key)
            entries.append(entry)
    return RSSDABank(entries=entries, version="runtime_multi_bank")


def build_retrieval_runtime(
    *,
    memory_bank: str | Path,
    bank_purpose: str,
    checkpoint: Optional[str],
    adapter_checkpoint: Optional[str],
    retriever_checkpoint: Optional[str],
    similarity_checkpoint: Optional[str],
    lora_checkpoint: Optional[str],
    lora_stage: str,
    device: str,
    precision: str,
    image_size: int,
    top_k: Optional[int],
    top_k_positive: Optional[int],
    top_k_negative: Optional[int],
    negative_lambda: float,
    positive_weight: float,
    negative_weight: float,
    similarity_threshold: float,
    confidence_scale: float,
    similarity_weighting: str,
    similarity_temperature: Optional[float],
    retrieval_policy: str,
    uncertainty_threshold: float,
    uncertainty_scale: float,
    policy_activation_threshold: float,
    residual_strength: float,
    allow_dummy_fallback: bool,
    continual_bank_root: Optional[str | Path] = None,
    site_bank_mode: str = "train_plus_site",
) -> RetrievalRuntime:
    """构建完整的检索运行时，包括模型加载、库加载和组件初始化。

    参数：
        - memory_bank: 记忆库路径
        - bank_purpose: 库用途
        - checkpoint: SAM3 检查点
        - adapter_checkpoint: 适配器检查点
        - retriever_checkpoint: 检索器检查点
        - similarity_checkpoint: 相似度模型检查点
        - lora_checkpoint: LoRA 检查点
        - lora_stage: LoRA 阶段
        - device: 设备
        - precision: 精度
        - image_size: 图像尺寸
        - top_k: 总检索数量
        - top_k_positive: 正例数量
        - top_k_negative: 负例数量
        - negative_lambda: 负例 lambda
        - positive_weight: 正例权重
        - negative_weight: 负例权重
        - similarity_threshold: 相似度阈值
        - confidence_scale: 置信度缩放
        - similarity_weighting: 相似度加权方式
        - similarity_temperature: 相似度温度
        - retrieval_policy: 检索策略
        - uncertainty_threshold: 不确定性阈值
        - uncertainty_scale: 不确定性缩放
        - policy_activation_threshold: 策略激活阈值
        - residual_strength: 残差强度
        - allow_dummy_fallback: 是否允许虚拟回退
        - continual_bank_root: 持续库根目录
        - site_bank_mode: 站点库模式

    返回：
        - RetrievalRuntime 实例
    """
    resolved_top_k_positive = int(top_k_positive or top_k or 1)
    resolved_top_k_negative = int(top_k_negative or top_k or 1)
    bank_context = load_retrieval_bank(
        memory_bank,
        purpose=bank_purpose,
        checkpoint=checkpoint,
        device=device,
        precision=precision,
        image_size=image_size,
        allow_dummy_fallback=allow_dummy_fallback,
    )
    if bank_purpose == "external-eval" and hasattr(bank_context.bank, "check_no_external_leakage"):
        if not bool(bank_context.bank.check_no_external_leakage(["PolypGen"])):
            raise RuntimeError("PolypGen leakage detected in retrieval bank for external evaluation.")
    base_model = build_official_sam3_image_model(
        checkpoint,
        device=device,
        dtype=precision,
        compile_model=False,
        allow_dummy_fallback=allow_dummy_fallback,
    )
    if lora_checkpoint and Path(lora_checkpoint).exists():
        apply_lora_to_model(base_model, LoRAConfig(stage=lora_stage, min_replaced_modules=0))
        load_lora_weights(base_model, lora_checkpoint, strict=False)
    freeze_model(base_model)
    wrapper = Sam3TensorForwardWrapper(model=base_model, device=device, dtype=precision)
    hidden_dim = resolve_hidden_dim(base_model)
    retriever = PrototypeRetriever(
        bank=bank_context.bank,
        feature_dim=hidden_dim,
        top_k_positive=resolved_top_k_positive,
        top_k_negative=resolved_top_k_negative,
    ).to(device)
    similarity_builder = SimilarityHeatmapBuilder(lambda_negative=negative_lambda).to(device)
    adapter = RetrievalSpatialSemanticAdapter(
        dim=hidden_dim,
        positive_weight=positive_weight,
        negative_weight=negative_weight,
        similarity_threshold=similarity_threshold,
        confidence_scale=confidence_scale,
        similarity_weighting=similarity_weighting,
        similarity_temperature=similarity_temperature,
        retrieval_policy=retrieval_policy,
        uncertainty_threshold=uncertainty_threshold,
        uncertainty_scale=uncertainty_scale,
        policy_activation_threshold=policy_activation_threshold,
        residual_strength=residual_strength,
    ).to(device)

    directory_loader = None
    if bank_context.source == "directory_bank":
        directory_loader = DirectoryBankLoader(
            bank_context.resolved_path,
            checkpoint=checkpoint,
            device=device,
            precision=precision,
            image_size=image_size,
            allow_dummy_fallback=allow_dummy_fallback,
            default_top_k=max(resolved_top_k_positive, resolved_top_k_negative),
        )

    loaded_components = {"adapter": False, "retriever": False, "similarity_builder": False}
    if adapter_checkpoint and Path(adapter_checkpoint).exists():
        loaded_components = load_rssda_bundle_components(
            adapter_checkpoint,
            device=device,
            adapter=adapter,
            retriever=retriever,
            similarity_builder=similarity_builder,
        )
        if not loaded_components["adapter"]:
            adapter.load_state_dict(torch.load(adapter_checkpoint, map_location=device, weights_only=False), strict=False)
            loaded_components["adapter"] = True
    if retriever_checkpoint and Path(retriever_checkpoint).exists():
        retriever.load_state_dict(torch.load(retriever_checkpoint, map_location=device, weights_only=False), strict=False)
        loaded_components["retriever"] = True
    if similarity_checkpoint and Path(similarity_checkpoint).exists():
        similarity_builder.load_state_dict(torch.load(similarity_checkpoint, map_location=device, weights_only=False), strict=False)
        loaded_components["similarity_builder"] = True

    adapter.eval()
    retriever.eval()
    similarity_builder.eval()
    retrieval_backend = "directory_loader" if directory_loader is not None and not loaded_components["retriever"] else "trainable_retriever"
    primary_backend = RetrievalBankBackend(
        name="train_bank",
        bank_context=bank_context,
        retriever=retriever,
        retrieval_backend=retrieval_backend,
        directory_loader=directory_loader,
    )
    inferred_continual_root = _resolve_site_bank_root(memory_bank, continual_bank_root)
    runtime = RetrievalRuntime(
        device=device,
        hidden_dim=hidden_dim,
        bank_context=bank_context,
        wrapper=wrapper,
        retriever=retriever,
        similarity_builder=similarity_builder,
        adapter=adapter,
        retrieval_backend=retrieval_backend,
        directory_loader=directory_loader,
        primary_backend=primary_backend,
        site_bank_mode=site_bank_mode,
        continual_bank_root=inferred_continual_root,
        bank_loader_config={
            "bank_purpose": bank_purpose,
            "checkpoint": checkpoint,
            "precision": precision,
            "image_size": image_size,
            "allow_dummy_fallback": allow_dummy_fallback,
            "top_k_positive": resolved_top_k_positive,
            "top_k_negative": resolved_top_k_negative,
        },
        bank_backend_cache={_backend_cache_key(bank_context.resolved_path): primary_backend},
    )
    return runtime


def resolve_retrieval(
    runtime: RetrievalRuntime,
    query_feature: torch.Tensor,
    *,
    top_k: Optional[int] = None,
    top_k_positive: Optional[int] = None,
    top_k_negative: Optional[int] = None,
    retrieval_mode: str,
    query_source: Optional[str] = None,
    prefer_cross_domain_positive: bool = True,
    sample_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """执行完整的检索解析流程：库选择、后端检索、标注、掩码先验附着、模式应用。

    参数：
        - runtime: 检索运行时
        - query_feature: 查询特征
        - top_k: 总检索数量
        - top_k_positive: 正例数量
        - top_k_negative: 负例数量
        - retrieval_mode: 检索模式
        - query_source: 查询来源域
        - prefer_cross_domain_positive: 是否优先选用跨域正例
        - sample_metadata: 样本元数据

    返回：
        - 完整的检索结果字典
    """
    resolution = _resolve_bank_selection(runtime, sample_metadata)
    backends = [_get_backend(runtime, bank_path) for bank_path in resolution.selected_bank_paths]
    if len(backends) <= 1:
        backend = backends[0] if backends else runtime.primary_backend
        if backend is None:
            raise RuntimeError("No retrieval backend available.")
        retrieval = _run_backend_retrieval(
            backend,
            query_feature,
            top_k=top_k,
            top_k_positive=top_k_positive,
            top_k_negative=top_k_negative,
            query_source=query_source,
            prefer_cross_domain_positive=prefer_cross_domain_positive,
        )
        retrieval = annotate_single_bank_retrieval(
            retrieval,
            resolution=resolution,
            bank_label="train" if backend.bank_context.resolved_path == runtime.bank_context.resolved_path else "site",
            bank_path=str(backend.bank_context.resolved_path),
        )
    else:
        train_backend = backends[0]
        site_backend = backends[1]
        train_retrieval = _run_backend_retrieval(
            train_backend,
            query_feature,
            top_k=top_k,
            top_k_positive=top_k_positive or top_k,
            top_k_negative=top_k_negative or top_k,
            query_source=query_source,
            prefer_cross_domain_positive=prefer_cross_domain_positive,
        )
        site_retrieval = _run_backend_retrieval(
            site_backend,
            query_feature,
            top_k=top_k,
            top_k_positive=top_k_positive or top_k,
            top_k_negative=top_k_negative or top_k,
            query_source=query_source,
            prefer_cross_domain_positive=prefer_cross_domain_positive,
        )
        retrieval = fuse_multi_bank_retrieval(
            train_retrieval=train_retrieval,
            site_retrieval=site_retrieval,
            resolution=resolution,
            train_bank_path=str(train_backend.bank_context.resolved_path),
            site_bank_path=str(site_backend.bank_context.resolved_path),
        )
    retrieval = attach_retrieved_mask_priors(retrieval, spatial_size=(int(query_feature.shape[-2]), int(query_feature.shape[-1])))
    return apply_retrieval_mode(retrieval, retrieval_mode)


def run_retrieval_forward(
    runtime: RetrievalRuntime,
    *,
    images: torch.Tensor,
    boxes: torch.Tensor,
    text_prompt: list[str],
    query_feature: torch.Tensor,
    retrieval: dict[str, Any],
    retrieval_mode: str,
    baseline_mask_logits: Optional[torch.Tensor] = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """执行包含检索的完整前向推理，返回输出、检索先验、适配器 Aux 及相似度。

    参数：
        - runtime: 检索运行时
        - images: 图像张量
        - boxes: 边界框张量
        - text_prompt: 文本提示
        - query_feature: 查询特征
        - retrieval: 检索结果
        - retrieval_mode: 检索模式
        - baseline_mask_logits: 基线掩码 logits（可选）

    返回：
        - (模型输出, 检索先验, 适配器辅助输出, 相似度字典) 的元组
    """
    similarity = runtime.similarity_builder(
        query_feature,
        retrieval["positive_features"],
        retrieval["negative_features"],
        retrieval["positive_weights"],
        retrieval["negative_weights"],
    )
    positive_heatmap = similarity["positive_heatmap"]
    negative_heatmap = similarity["negative_heatmap"]
    if isinstance(retrieval.get("positive_mask_prior"), torch.Tensor):
        positive_heatmap = torch.clamp(0.7 * positive_heatmap + 0.3 * retrieval["positive_mask_prior"].to(positive_heatmap), 0.0, 1.0)
    if isinstance(retrieval.get("negative_mask_prior"), torch.Tensor):
        negative_heatmap = torch.clamp(0.7 * negative_heatmap + 0.3 * retrieval["negative_mask_prior"].to(negative_heatmap), 0.0, 1.0)
    _, retrieval_prior, adapter_aux = runtime.adapter(
        feature_map=query_feature,
        similarity_map=similarity["fused_similarity"],
        positive_prototype=retrieval["positive_prototype"],
        negative_prototype=retrieval["negative_prototype"],
        positive_tokens=retrieval["positive_features"],
        negative_tokens=retrieval["negative_features"],
        positive_similarity=similarity["positive_similarity"],
        negative_similarity=similarity["negative_similarity"],
        positive_weights=retrieval["positive_weights"],
        negative_weights=retrieval["negative_weights"],
        positive_scores=retrieval.get("positive_score_tensor"),
        negative_scores=retrieval.get("negative_score_tensor"),
        baseline_mask_logits=baseline_mask_logits,
        positive_heatmap=positive_heatmap,
        negative_heatmap=negative_heatmap,
        mode=retrieval_mode,
    )
    outputs = runtime.wrapper(images=images, boxes=boxes, text_prompt=text_prompt, retrieval_prior=retrieval_prior)
    return outputs, retrieval_prior, adapter_aux, similarity


def infer_query_feature(runtime: RetrievalRuntime, images: torch.Tensor, boxes: torch.Tensor, text_prompt: list[str]) -> tuple[dict[str, Any], torch.Tensor]:
    """运行基线推理并解析查询特征用于后续检索。

    参数：
        - runtime: 检索运行时
        - images: 图像张量
        - boxes: 边界框张量
        - text_prompt: 文本提示

    返回：
        - (基线输出, 查询特征张量) 的元组
    """
    baseline = runtime.wrapper(images=images, boxes=boxes, text_prompt=text_prompt)
    return baseline, resolve_feature_map(baseline["image_embeddings"], images)
