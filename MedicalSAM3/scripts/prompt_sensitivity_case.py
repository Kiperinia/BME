"""Run same-image different-exemplar prompt sensitivity experiments."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from PIL import Image
import torch

from MedicalSAM3.evaluation.retrieval_analysis import (
    _empty_negative_like,
    _prompt_sensitivity,
    _rank_entry_indices,
    _selection_from_entries,
)
from MedicalSAM3.evaluation.retrieval_influence import _mask_difference_ratio
from MedicalSAM3.scripts.common import apply_config_overrides, compute_segmentation_metrics, ensure_dir, infer_source_domain, load_config, log_runtime_environment, resolve_runtime_device
from MedicalSAM3.scripts.retrieval_runtime import (
    build_retrieval_runtime,
    collect_input_images,
    infer_query_feature,
    load_bbox_mapping,
    load_image_tensor,
    parse_bbox,
    resolve_effective_bank,
    resolve_retrieval,
    run_retrieval_forward,
    scale_bbox,
)


def _load_path_mapping(path: str | Path) -> dict[str, str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return {str(key): str(value) for key, value in payload.items()}
    if isinstance(payload, list):
        mapping: dict[str, str] = {}
        for item in payload:
            if not isinstance(item, dict):
                continue
            key = str(item.get("image") or item.get("image_id") or item.get("path") or "")
            value = item.get("mask_path") or item.get("mask")
            if key and value:
                mapping[key] = str(value)
        return mapping
    raise ValueError(f"Unsupported path mapping payload: {path}")


def _resolve_bbox_for_image(image_path: Path, bbox_literal: str | None, bbox_mapping: dict[str, list[float]]) -> list[float]:
    if bbox_literal:
        return parse_bbox(bbox_literal)
    for key in [image_path.name, image_path.stem, image_path.as_posix()]:
        if key in bbox_mapping:
            return bbox_mapping[key]
    raise ValueError(f"No bbox provided for {image_path}")


def _load_mask(mask_path: str | None, image_size: int) -> torch.Tensor | None:
    if not mask_path:
        return None
    target = Path(mask_path)
    if not target.exists():
        return None
    mask = Image.open(target).convert("L").resize((image_size, image_size), resample=Image.NEAREST)
    mask_array = np.asarray(mask)
    threshold = 0 if mask_array.max() <= 1 else 127
    return torch.from_numpy((mask_array > threshold).astype("float32")).unsqueeze(0).unsqueeze(0)


def _overlay_image(image: Image.Image, mask_logits: torch.Tensor) -> Image.Image:
    mask = (torch.sigmoid(mask_logits[0, 0]).detach().cpu().numpy() > 0.5).astype(np.float32)
    mask = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).resize(image.size, resample=Image.NEAREST)).astype(np.float32) / 255.0
    base = np.asarray(image).astype("float32")
    overlay = base.copy()
    overlay[mask > 0.5] = 0.65 * overlay[mask > 0.5] + 0.35 * np.array([255.0, 64.0, 64.0], dtype=np.float32)
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


def _save_influence_heatmap(path: Path, current_logits: torch.Tensor, baseline_logits: torch.Tensor) -> None:
    diff = (torch.sigmoid(current_logits[0, 0]).detach().cpu().float() - torch.sigmoid(baseline_logits[0, 0]).detach().cpu().float()).abs().numpy()
    diff = diff - diff.min()
    denom = max(float(diff.max()), 1e-6)
    norm = diff / denom
    rgb = np.stack([norm, np.sqrt(norm), 1.0 - norm], axis=-1)
    Image.fromarray((rgb * 255.0).clip(0, 255).astype(np.uint8)).save(path)


def _empty_positive_like(retrieval: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, list[Any], torch.Tensor, torch.Tensor]:
    dim = int(retrieval["negative_prototype"].shape[-1])
    device = retrieval["negative_prototype"].device
    return (
        torch.zeros(1, 0, dim, device=device),
        torch.zeros(1, 0, device=device),
        [],
        torch.zeros(1, dim, device=device),
        torch.zeros(0, device=device),
    )


def _override_retrieval(
    base_retrieval: dict[str, Any],
    *,
    positive_override: tuple[torch.Tensor, torch.Tensor, list[Any], torch.Tensor, torch.Tensor] | None = None,
    negative_override: tuple[torch.Tensor, torch.Tensor, list[Any], torch.Tensor, torch.Tensor] | None = None,
) -> dict[str, Any]:
    retrieval = dict(base_retrieval)
    if positive_override is not None:
        retrieval["positive_features"] = positive_override[0]
        retrieval["positive_weights"] = positive_override[1]
        retrieval["positive_entries"] = [positive_override[2]]
        retrieval["positive_prototype"] = positive_override[3]
        retrieval["positive_scores"] = [positive_override[4]]
        retrieval["positive_score_tensor"] = positive_override[4].unsqueeze(0) if positive_override[4].numel() > 0 else torch.zeros(1, 0, device=positive_override[3].device)
    if negative_override is not None:
        retrieval["negative_features"] = negative_override[0]
        retrieval["negative_weights"] = negative_override[1]
        retrieval["negative_entries"] = [negative_override[2]]
        retrieval["negative_prototype"] = negative_override[3]
        retrieval["negative_scores"] = [negative_override[4]]
        retrieval["negative_score_tensor"] = negative_override[4].unsqueeze(0) if negative_override[4].numel() > 0 else torch.zeros(1, 0, device=negative_override[3].device)
    return retrieval


def _build_variants(
    base_retrieval: dict[str, Any],
    *,
    bank: Any,
    query_vector: torch.Tensor,
    top_k_positive: int,
    top_k_negative: int,
    seed: int,
) -> dict[str, Any | None]:
    positive_entries = bank.get_entries(polarity="positive", human_verified=True)
    negative_entries = bank.get_entries(polarity="negative", human_verified=True)
    rng = random.Random(seed)
    random_positive = _selection_from_entries(
        bank,
        query_vector,
        positive_entries,
        _rank_entry_indices(bank, query_vector, positive_entries, top_k_positive, "random", rng),
    )
    return {
        "positive_retrieval": _override_retrieval(
            base_retrieval,
            negative_override=_empty_negative_like(base_retrieval),
        ),
        "negative_retrieval": _override_retrieval(
            base_retrieval,
            positive_override=_empty_positive_like(base_retrieval),
        ),
        "random_retrieval": _override_retrieval(
            base_retrieval,
            positive_override=random_positive,
            negative_override=_empty_negative_like(base_retrieval),
        ),
        "no_retrieval": None,
    }


def _run_variant(
    runtime: Any,
    *,
    images: torch.Tensor,
    boxes: torch.Tensor,
    text_prompt: list[str],
    query_feature: torch.Tensor,
    baseline_outputs: dict[str, Any],
    retrieval: dict[str, Any] | None,
    retrieval_mode: str,
    mask: torch.Tensor | None,
) -> dict[str, Any]:
    if retrieval is None:
        outputs = baseline_outputs
        adapter_aux: dict[str, Any] = {}
    else:
        outputs, _, adapter_aux, _ = run_retrieval_forward(
            runtime,
            images=images,
            boxes=boxes,
            text_prompt=text_prompt,
            query_feature=query_feature,
            retrieval=retrieval,
            retrieval_mode=retrieval_mode,
            baseline_mask_logits=baseline_outputs.get("mask_logits"),
        )
    metrics = compute_segmentation_metrics(outputs["mask_logits"], mask.to(outputs["mask_logits"].device)) if mask is not None else {"Dice": 0.0, "IoU": 0.0}
    return {
        "mask_logits": outputs["mask_logits"].detach(),
        "metrics": metrics,
        "policy_diagnostics": {
            "retrieval_activation_ratio": float(adapter_aux.get("retrieval_activation_ratio", torch.tensor(0.0)).detach().float().mean().item()) if adapter_aux else 0.0,
            "retrieval_suppression_ratio": float(adapter_aux.get("retrieval_suppression_ratio", torch.tensor(0.0)).detach().float().mean().item()) if adapter_aux else 1.0,
            "uncertainty_gate": float(adapter_aux.get("uncertainty_gate", torch.tensor(0.0)).detach().float().mean().item()) if adapter_aux else 0.0,
            "segmentation_confidence": float(adapter_aux.get("segmentation_confidence", torch.tensor(0.0)).detach().float().mean().item()) if adapter_aux else float((torch.sigmoid(outputs["mask_logits"]) - 0.5).abs().mul(2.0).mean().item()),
            "segmentation_uncertainty": float(adapter_aux.get("segmentation_uncertainty", torch.tensor(0.0)).detach().float().mean().item()) if adapter_aux else float((1.0 - (torch.sigmoid(outputs["mask_logits"]) - 0.5).abs().mul(2.0)).mean().item()),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run prompt sensitivity experiments for same-image different-exemplar variants.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--bbox", default=None)
    parser.add_argument("--bbox-json", default=None)
    parser.add_argument("--mask-path", default=None)
    parser.add_argument("--mask-json", default=None)
    parser.add_argument("--memory-bank", default="MedicalSAM3/banks/train_bank")
    parser.add_argument("--continual-bank-root", default=None)
    parser.add_argument("--bank-purpose", default="external-eval", choices=["external-eval", "validation", "train", "continual-adaptation"])
    parser.add_argument("--site-bank-mode", default="train_plus_site", choices=["train_only", "site_only", "train_plus_site"])
    parser.add_argument("--output-dir", default="MedicalSAM3/outputs/medex_sam3/prompt_sensitivity")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--adapter-checkpoint", default=None)
    parser.add_argument("--retriever-checkpoint", default=None)
    parser.add_argument("--similarity-checkpoint", default=None)
    parser.add_argument("--lora-checkpoint", default=None)
    parser.add_argument("--lora-stage", default="stage_a")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--precision", default="fp32")
    parser.add_argument("--top-k-positive", type=int, default=1)
    parser.add_argument("--top-k-negative", type=int, default=1)
    parser.add_argument("--negative-lambda", type=float, default=0.35)
    parser.add_argument("--positive-weight", type=float, default=1.0)
    parser.add_argument("--negative-weight", type=float, default=0.25)
    parser.add_argument("--similarity-threshold", type=float, default=0.5)
    parser.add_argument("--confidence-scale", type=float, default=8.0)
    parser.add_argument("--similarity-weighting", choices=["hard", "soft"], default="soft")
    parser.add_argument("--similarity-temperature", type=float, default=0.125)
    parser.add_argument("--retrieval-policy", choices=["always-on", "similarity-threshold", "uncertainty-aware", "region-aware", "residual"], default="uncertainty-aware")
    parser.add_argument("--uncertainty-threshold", type=float, default=0.35)
    parser.add_argument("--uncertainty-scale", type=float, default=10.0)
    parser.add_argument("--policy-activation-threshold", type=float, default=0.05)
    parser.add_argument("--residual-strength", type=float, default=0.5)
    parser.add_argument("--retrieval-mode", choices=["joint", "semantic", "spatial", "positive-only", "negative-only", "positive-negative"], default="joint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dummy", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    apply_config_overrides(
        args,
        config,
        {
            "top_k_positive": 1,
            "top_k_negative": 1,
            "negative_lambda": 0.35,
            "positive_weight": 1.0,
            "negative_weight": 0.25,
            "similarity_threshold": 0.5,
            "confidence_scale": 8.0,
            "similarity_weighting": "soft",
            "similarity_temperature": 0.125,
            "retrieval_policy": "uncertainty-aware",
            "uncertainty_threshold": 0.35,
            "uncertainty_scale": 10.0,
            "policy_activation_threshold": 0.05,
            "residual_strength": 0.5,
            "retrieval_mode": "joint",
        },
    )

    device = resolve_runtime_device(args.device)
    log_runtime_environment(
        "prompt_sensitivity_case",
        requested_device=args.device,
        resolved_device=device,
        extra={"dummy": bool(args.dummy), "image_size": int(args.image_size)},
    )
    bbox_mapping = load_bbox_mapping(args.bbox_json) if args.bbox_json else {}
    mask_mapping = _load_path_mapping(args.mask_json) if args.mask_json else {}
    output_dir = ensure_dir(args.output_dir)
    runtime = build_retrieval_runtime(
        memory_bank=args.memory_bank,
        continual_bank_root=args.continual_bank_root,
        bank_purpose=args.bank_purpose,
        checkpoint=args.checkpoint,
        adapter_checkpoint=args.adapter_checkpoint,
        retriever_checkpoint=args.retriever_checkpoint,
        similarity_checkpoint=args.similarity_checkpoint,
        lora_checkpoint=args.lora_checkpoint,
        lora_stage=args.lora_stage,
        device=device,
        precision=args.precision,
        image_size=args.image_size,
        top_k=None,
        top_k_positive=args.top_k_positive,
        top_k_negative=args.top_k_negative,
        negative_lambda=args.negative_lambda,
        positive_weight=args.positive_weight,
        negative_weight=args.negative_weight,
        similarity_threshold=args.similarity_threshold,
        confidence_scale=args.confidence_scale,
        similarity_weighting=args.similarity_weighting,
        similarity_temperature=args.similarity_temperature,
        retrieval_policy=args.retrieval_policy,
        site_bank_mode=args.site_bank_mode,
        uncertainty_threshold=args.uncertainty_threshold,
        uncertainty_scale=args.uncertainty_scale,
        policy_activation_threshold=args.policy_activation_threshold,
        residual_strength=args.residual_strength,
        allow_dummy_fallback=args.dummy,
    )

    rows = []
    input_images = collect_input_images(args.input_path)
    for index, image_path in enumerate(input_images):
        print(json.dumps({"progress": {"script": "prompt_sensitivity_case", "index": index + 1, "total": len(input_images), "device": device}}, ensure_ascii=True), flush=True)
        original_image, image_tensor = load_image_tensor(image_path, args.image_size)
        bbox = _resolve_bbox_for_image(image_path, args.bbox, bbox_mapping)
        scaled_box = scale_bbox(bbox, original_image.size, args.image_size).unsqueeze(0).to(device)
        images = image_tensor.to(device)
        text_prompt = ["polyp"]
        baseline_outputs, query_feature = infer_query_feature(runtime, images=images, boxes=scaled_box, text_prompt=text_prompt)
        query_source = infer_source_domain(image_path=str(image_path), image_id=image_path.stem, dataset_name=image_path.parent.name)
        sample_metadata = {
            "image_path": str(image_path),
            "image_id": image_path.stem,
            "dataset_name": "PolypGen" if "polypgen" in image_path.as_posix().lower() else image_path.parent.name,
        }
        base_retrieval = resolve_retrieval(
            runtime,
            query_feature,
            top_k_positive=args.top_k_positive,
            top_k_negative=args.top_k_negative,
            retrieval_mode=args.retrieval_mode,
            query_source=query_source,
            prefer_cross_domain_positive=True,
            sample_metadata=sample_metadata,
        )
        query_vector = base_retrieval["projected_query"][0]
        variants_retrieval = _build_variants(
            base_retrieval,
            bank=resolve_effective_bank(runtime, sample_metadata=sample_metadata),
            query_vector=query_vector,
            top_k_positive=args.top_k_positive,
            top_k_negative=args.top_k_negative,
            seed=args.seed + index,
        )
        mask_path = (
            args.mask_path
            or mask_mapping.get(image_path.name)
            or mask_mapping.get(image_path.stem)
            or mask_mapping.get(image_path.as_posix())
        )
        mask = _load_mask(mask_path, args.image_size)

        variants = {
            name: _run_variant(
                runtime,
                images=images,
                boxes=scaled_box,
                text_prompt=text_prompt,
                query_feature=query_feature,
                baseline_outputs=baseline_outputs,
                retrieval=retrieval,
                retrieval_mode=args.retrieval_mode,
                mask=mask,
            )
            for name, retrieval in variants_retrieval.items()
        }
        legacy_variants = {
            "positive_exemplar": variants["positive_retrieval"],
            "negative_exemplar": variants["negative_retrieval"],
            "random_exemplar": variants["random_retrieval"],
            "empty_exemplar": variants["no_retrieval"],
        }
        summary = _prompt_sensitivity(legacy_variants)
        summary["dice_variance_available"] = mask is not None
        case_dir = ensure_dir(output_dir / image_path.stem)
        baseline_logits = variants["no_retrieval"]["mask_logits"]
        variant_rows = {}
        for variant_name, payload in variants.items():
            _overlay_image(original_image, payload["mask_logits"]).save(case_dir / f"{variant_name}_overlay.png")
            heatmap_path = None
            if variant_name != "no_retrieval":
                heatmap_path = case_dir / f"{variant_name}_influence_heatmap.png"
                _save_influence_heatmap(heatmap_path, payload["mask_logits"], baseline_logits)
            variant_rows[variant_name] = {
                "metrics": payload["metrics"],
                "mask_difference_ratio": 0.0 if variant_name == "no_retrieval" else _mask_difference_ratio(payload["mask_logits"], baseline_logits),
                "dice_delta": float(payload["metrics"].get("Dice", 0.0) - variants["no_retrieval"]["metrics"].get("Dice", 0.0)),
                "boundary_delta": float(payload["metrics"].get("Boundary F1", 0.0) - variants["no_retrieval"]["metrics"].get("Boundary F1", 0.0)),
                "policy_diagnostics": payload["policy_diagnostics"],
                "retrieval_influence_heatmap": None if heatmap_path is None else str(heatmap_path),
            }
        row = {
            "image_id": image_path.stem,
            "image_path": str(image_path),
            "mask_path": mask_path,
            "baseline_confidence": float((torch.sigmoid(baseline_outputs["mask_logits"]) - 0.5).abs().mul(2.0).mean().item()),
            "baseline_uncertainty": float((1.0 - (torch.sigmoid(baseline_outputs["mask_logits"]) - 0.5).abs().mul(2.0)).mean().item()),
            "prompt_sensitivity": summary,
            "same_image_variants": variant_rows,
            "visualization_dir": str(case_dir),
        }
        rows.append(row)

    (output_dir / "prompt_sensitivity.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    overall_score = float(sum(row["prompt_sensitivity"]["prompt_sensitivity_score"] for row in rows) / max(len(rows), 1))
    sorted_by_confidence = sorted(rows, key=lambda row: row["baseline_confidence"])
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "count": len(rows),
                "overall_prompt_sensitivity_score": overall_score,
                "confidence_case_summary": {
                    "lowest_confidence_case": None if not sorted_by_confidence else {
                        "image_id": sorted_by_confidence[0]["image_id"],
                        "baseline_confidence": sorted_by_confidence[0]["baseline_confidence"],
                        "baseline_uncertainty": sorted_by_confidence[0]["baseline_uncertainty"],
                    },
                    "highest_confidence_case": None if not sorted_by_confidence else {
                        "image_id": sorted_by_confidence[-1]["image_id"],
                        "baseline_confidence": sorted_by_confidence[-1]["baseline_confidence"],
                        "baseline_uncertainty": sorted_by_confidence[-1]["baseline_uncertainty"],
                    },
                },
                "variant_delta_summary": {
                    variant_name: {
                        "mean_mask_difference_ratio": float(sum(row["same_image_variants"][variant_name]["mask_difference_ratio"] for row in rows) / max(len(rows), 1)),
                        "mean_dice_delta": float(sum(row["same_image_variants"][variant_name]["dice_delta"] for row in rows) / max(len(rows), 1)),
                        "mean_boundary_delta": float(sum(row["same_image_variants"][variant_name]["boundary_delta"] for row in rows) / max(len(rows), 1)),
                        "mean_activation_ratio": float(sum(row["same_image_variants"][variant_name]["policy_diagnostics"]["retrieval_activation_ratio"] for row in rows) / max(len(rows), 1)),
                        "mean_suppression_ratio": float(sum(row["same_image_variants"][variant_name]["policy_diagnostics"]["retrieval_suppression_ratio"] for row in rows) / max(len(rows), 1)),
                    }
                    for variant_name in ["positive_retrieval", "negative_retrieval", "random_retrieval"]
                },
                "artifacts": {
                    "prompt_sensitivity": str(output_dir / "prompt_sensitivity.jsonl"),
                    "visualizations": str(output_dir),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"count": len(rows), "output_dir": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())