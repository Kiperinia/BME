"""
数据集加载器。

当前仅保留 KvasirCVC 与 PolypGen external test 的适配，以及通用医学分割数据集接口。
"""

import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Optional, Tuple, List, Dict, Any, Union

from utils.transforms import (
    get_train_transforms, get_val_transforms,
    mask_to_bbox, jitter_bbox,
)
from models.medsam3_base import DATASET_TEXT_PROMPTS


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")
ImageSource = Union[str, Tuple[str, ...]]
POLYPGEN_EXTERNAL_PREFIXES = ("C1", "C2", "C3", "C4", "C5", "C6")


class MedicalSegDataset(Dataset):
    """
    通用医学图像分割数据集
    支持 image + mask + bbox 的加载与增强。
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        transform: Any = None,
        image_size: int = 1024,
        bbox_json: Optional[str] = None,
        prompt_type: str = "bbox",
        jitter_bbox_ratio: float = 0.0,
        text_prompt: str = "",
    ):
        """
        Args:
            image_dir: 图像目录路径
            mask_dir: mask 目录路径
            transform: albumentations 变换
            image_size: 输出图像尺寸
            bbox_json: (可选) bbox 标注 json 文件路径
            prompt_type: 提示类型 (bbox / point)
            jitter_bbox_ratio: bbox 扰动比例 (0 表示不扰动)
            text_prompt: 文本提示词 (如 "polyp")
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_size = image_size
        self.prompt_type = prompt_type
        self.jitter_bbox_ratio = jitter_bbox_ratio
        self.text_prompt = text_prompt

        # 收集图像-mask 对
        self.samples = self._collect_samples()

        # 加载预计算的 bbox（如果有）
        self.bboxes: Dict[str, Any] = {}
        if bbox_json and os.path.isfile(bbox_json):
            with open(bbox_json, "r") as f:
                self.bboxes = json.load(f)

    def _collect_samples(self) -> List[Tuple[ImageSource, str]]:
        """收集匹配的 image-mask 文件对"""
        img_files = sorted(os.listdir(self.image_dir))
        mask_files = set(os.listdir(self.mask_dir))

        samples = []
        for img_name in img_files:
            stem = os.path.splitext(img_name)[0]
            # 尝试匹配 mask 文件 (支持多种命名约定)
            mask_name = None
            for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
                candidates = [
                    img_name,                      # 同名
                    stem + ext,                     # 同 stem 不同后缀
                    stem + "_mask" + ext,           # stem_mask
                ]
                for c in candidates:
                    if c in mask_files:
                        mask_name = c
                        break
                if mask_name:
                    break

            if mask_name:
                samples.append((
                    os.path.join(self.image_dir, img_name),
                    os.path.join(self.mask_dir, mask_name),
                ))

        return samples

    def _load_image(self, img_source: ImageSource) -> np.ndarray:
        """读取单张图像或多通道拆分图像。"""
        if isinstance(img_source, (tuple, list)):
            channels: List[np.ndarray] = []
            for channel_path in img_source:
                channel = cv2.imread(channel_path, cv2.IMREAD_GRAYSCALE)
                if channel is None:
                    raise FileNotFoundError(f"无法读取图像: {channel_path}")
                channels.append(channel)
            return np.stack(channels, axis=-1)

        image = cv2.imread(img_source)
        if image is None:
            raise FileNotFoundError(f"无法读取图像: {img_source}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _get_primary_image_path(self, img_source: ImageSource) -> str:
        if isinstance(img_source, (tuple, list)):
            return img_source[0]
        return img_source

    def _get_sample_stem(self, img_source: ImageSource) -> str:
        primary_path = self._get_primary_image_path(img_source)
        return os.path.splitext(os.path.basename(primary_path))[0]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_source, mask_path = self.samples[idx]

        # 读取图像 (BGR -> RGB)
        image = self._load_image(img_source)
        orig_h, orig_w = image.shape[:2]

        # 读取 mask (灰度, 二值化)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        mask = (mask > 127).astype(np.uint8)

        # 数据增强
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        # mask 后处理
        mask = mask.astype(np.float32)

        # 生成 prompt (bbox)
        bbox = self._get_bbox(img_source, mask, orig_h, orig_w)

        # numpy -> tensor
        if image.ndim == 3 and image.shape[-1] == 3:
            image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
        image_tensor = torch.from_numpy(image).float()
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)  # (1,H,W)
        bbox_tensor = torch.from_numpy(bbox).float()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "bbox": bbox_tensor,
            "image_path": self._get_primary_image_path(img_source),
            "text_prompt": self.text_prompt,
        }

    def _get_bbox(self, img_source: ImageSource, mask: np.ndarray,
                  orig_h: int, orig_w: int) -> np.ndarray:
        """获取 bounding box prompt"""
        stem = self._get_sample_stem(img_source)

        # 优先使用预计算的 bbox
        if stem in self.bboxes:
            info = self.bboxes[stem]
            b = info["bbox"][0] if isinstance(info["bbox"], list) and len(info["bbox"]) > 0 else info["bbox"]
            if isinstance(b, dict):
                bbox = np.array([b["xmin"], b["ymin"], b["xmax"], b["ymax"]], dtype=np.float32)
            else:
                bbox = np.array(b[:4], dtype=np.float32)
            # 缩放到当前 image_size
            scale_x = self.image_size / info.get("width", orig_w)
            scale_y = self.image_size / info.get("height", orig_h)
            bbox[0] *= scale_x
            bbox[2] *= scale_x
            bbox[1] *= scale_y
            bbox[3] *= scale_y
        else:
            # 从 mask 提取 bbox
            bbox = mask_to_bbox(mask)

        # 训练时扰动 bbox
        if self.jitter_bbox_ratio > 0:
            bbox = jitter_bbox(bbox, self.jitter_bbox_ratio,
                               self.image_size, self.image_size)

        return bbox


class NnUNetRawRGBDataset(MedicalSegDataset):
    """适配 nnUNet raw 三通道图像目录。"""

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        transform: Any = None,
        image_size: int = 1024,
        accepted_prefixes: Optional[Tuple[str, ...]] = None,
        **kwargs,
    ):
        self.accepted_prefixes = accepted_prefixes
        super().__init__(
            image_dir=image_dir,
            mask_dir=mask_dir,
            transform=transform,
            image_size=image_size,
            **kwargs,
        )

    def _collect_samples(self) -> List[Tuple[ImageSource, str]]:
        image_files = {
            name for name in os.listdir(self.image_dir)
            if os.path.isfile(os.path.join(self.image_dir, name))
        }
        mask_files = [
            name for name in sorted(os.listdir(self.mask_dir))
            if os.path.isfile(os.path.join(self.mask_dir, name))
            and os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
        ]

        samples: List[Tuple[ImageSource, str]] = []
        for mask_name in mask_files:
            stem, mask_ext = os.path.splitext(mask_name)
            if self.accepted_prefixes and not stem.startswith(self.accepted_prefixes):
                continue

            channel_paths: List[str] = []
            for channel_idx in range(3):
                channel_path = None
                candidate_exts = (mask_ext,) + tuple(
                    ext for ext in IMAGE_EXTENSIONS if ext != mask_ext
                )
                for ext in candidate_exts:
                    img_name = f"{stem}_{channel_idx:04d}{ext}"
                    if img_name in image_files:
                        channel_path = os.path.join(self.image_dir, img_name)
                        break
                if channel_path is None:
                    channel_paths = []
                    break
                channel_paths.append(channel_path)

            if channel_paths:
                samples.append((tuple(channel_paths), os.path.join(self.mask_dir, mask_name)))

        return samples

    def _get_sample_stem(self, img_source: ImageSource) -> str:
        stem = super()._get_sample_stem(img_source)
        if stem.endswith("_0000"):
            return stem[:-5]
        return stem


class KvasirCVCDataset(NnUNetRawRGBDataset):
    """KvasirCVC nnUNet raw 数据集。"""

    def __init__(self, data_root: str, transform: Any = None,
                 image_size: int = 1024, **kwargs):
        dataset_dir = os.path.join(
            data_root, "KvasirCVC-nnunet_raw", "Dataset504_KvasirCVC"
        )
        super().__init__(
            image_dir=os.path.join(dataset_dir, "imagesTr"),
            mask_dir=os.path.join(dataset_dir, "labelsTr"),
            transform=transform,
            image_size=image_size,
            text_prompt=DATASET_TEXT_PROMPTS.get("kvasircvc", "polyp"),
            **kwargs,
        )


class PolypGenDataset(NnUNetRawRGBDataset):
    """PolypGen external test 数据集，保留 C1-C6。"""

    def __init__(self, data_root: str, transform: Any = None,
                 image_size: int = 1024, **kwargs):
        dataset_dir = os.path.join(
            data_root, "PolypGen_external_test", "Dataset502_PolypGen"
        )
        super().__init__(
            image_dir=os.path.join(dataset_dir, "imagesTs"),
            mask_dir=os.path.join(dataset_dir, "labelsTs"),
            transform=transform,
            image_size=image_size,
            accepted_prefixes=POLYPGEN_EXTERNAL_PREFIXES,
            text_prompt=DATASET_TEXT_PROMPTS.get("polypgen", "polyp"),
            **kwargs,
        )


def create_dataset(
    dataset_name: str,
    data_root: str,
    transform: Any = None,
    image_size: int = 1024,
    **kwargs,
):
    """根据数据集名称创建实例，统一脚本入口与底层适配逻辑。"""

    normalized_name = dataset_name.lower()
    if normalized_name in {"kvasircvc", "kvasir-cvc", "kvasir_cvc", "dataset504_kvasircvc"}:
        return KvasirCVCDataset(
            data_root,
            transform=transform,
            image_size=image_size,
            **kwargs,
        )
    if normalized_name in {"polypgen", "polypgen_external_test", "dataset502_polypgen"}:
        return PolypGenDataset(
            data_root,
            transform=transform,
            image_size=image_size,
            **kwargs,
        )
    raise ValueError(f"不支持的数据集: {dataset_name}")


def build_dataloaders(
    dataset_name: str,
    data_root: str,
    image_size: int = 1024,
    batch_size: int = 4,
    train_ratio: float = 0.85,
    num_workers: int = 4,
    seed: int = 42,
    jitter_bbox_ratio: float = 0.05,
) -> Tuple[DataLoader, DataLoader]:
    """
    构建训练和验证 DataLoader
    """
    train_tf = get_train_transforms(image_size)
    val_tf = get_val_transforms(image_size)

    full_dataset = create_dataset(
        dataset_name,
        data_root,
        transform=None,
        image_size=image_size,
        jitter_bbox_ratio=jitter_bbox_ratio,
    )

    # 划分训练/验证集
    n_total = len(full_dataset)
    n_train = int(n_total * train_ratio)
    n_val = n_total - n_train

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        full_dataset, [n_train, n_val], generator=generator
    )

    # 为 subset 分别设置不同的 transform
    train_dataset = TransformSubset(train_subset, train_tf)
    val_dataset = TransformSubset(val_subset, val_tf)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


class TransformSubset(Dataset):
    """对 Subset 施加独立的 transform"""

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        sample = self.subset[idx]
        # 如果 sample 是 dict，对 image 和 mask 重新做 transform
        if isinstance(sample, dict) and self.transform is not None:
            # 先把 tensor 还原为 numpy (CHW -> HWC)
            img = sample["image"]
            msk = sample["mask"]
            if isinstance(img, torch.Tensor):
                if img.dim() == 3 and img.shape[0] == 3:
                    img = img.permute(1, 2, 0).numpy()
                else:
                    img = img.numpy()
            if isinstance(msk, torch.Tensor):
                msk = msk.squeeze(0).numpy()

            # 反归一化 (如果已经归一化了)
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
            msk = msk.astype(np.uint8)

            transformed = self.transform(image=img, mask=msk)
            t_img = transformed["image"]
            t_msk = transformed["mask"]

            if t_img.ndim == 3 and t_img.shape[-1] == 3:
                t_img = np.transpose(t_img, (2, 0, 1))

            sample = {
                **sample,
                "image": torch.from_numpy(t_img).float(),
                "mask": torch.from_numpy(t_msk.astype(np.float32)).unsqueeze(0),
            }
        return sample
