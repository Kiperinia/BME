"""Run retrieval-conditioned inference for a single image or a folder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from MedicalSAM3.evaluation.retrieval_diagnostics import build_retrieval_diagnostics, summarize_retrieval_diagnostics, write_retrieval_diagnostics
from MedicalSAM3.scripts.common import apply_config_overrides, ensure_dir, infer_source_domain, load_config, log_runtime_environment, resolve_runtime_device
from MedicalSAM3.scripts.retrieval_runtime import (
    build_retrieval_runtime,
    collect_input_images,
    infer_query_feature,
    load_bbox_mapping,
    load_image_tensor,
    parse_bbox,
    resolve_retrieval,
    run_retrieval_forward,
    scale_bbox,
)


def _overlay_image(image: Image.Image, mask: np.ndarray) -> Image.Image:
    base = np.asarray(image).astype("float32")
    overlay = base.copy()
    overlay[mask > 0.5] = 0.65 * overlay[mask > 0.5] + 0.35 * np.array([255.0, 64.0, 64.0], dtype=np.float32)
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


def _save_preview(
    positive_entries: list[dict[str, Any]],
    negative_entries: list[dict[str, Any]],
    output_path: Path,
    image_size: int = 128,
) -> None:
    preview_paths = [entry.get("crop_path") for entry in positive_entries + negative_entries if entry.get("crop_path")]
    images = []
    for path in preview_paths:
        candidate = Path(str(path))
        if candidate.exists():
            images.append(Image.open(candidate).convert("RGB").resize((image_size, image_size)))
    if not images:
        images = [Image.new("RGB", (image_size, image_size), color=(230, 230, 230))]
    panel = Image.new("RGB", (image_size * len(images), image_size), color=(255, 255, 255))
    for index, image in enumerate(images):
        panel.paste(image, (index * image_size, 0))
    panel.save(output_path)


def _resolve_bbox_for_image(
    image_path: Path,
    bbox_literal: str | None,
    bbox_mapping: dict[str, list[float]],
) -> list[float]:
    if bbox_literal:
        return parse_bbox(bbox_literal)
    for key in [image_path.name, image_path.stem, image_path.as_posix()]:
        if key in bbox_mapping:
            return bbox_mapping[key]
    raise ValueError(f"No bbox provided for {image_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run retrieval-conditioned inference.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--bbox", default=None)
    parser.add_argument("--bbox-json", default=None)
    parser.add_argument("--memory-bank", default="MedicalSAM3/banks/train_bank")
    parser.add_argument("--bank-purpose", default="external-eval", choices=["external-eval", "validation", "train", "continual-adaptation"])
    parser.add_argument("--output-dir", default="MedicalSAM3/outputs/medex_sam3/retrieval_inference")
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
        "run_retrieval_inference",
        requested_device=args.device,
        resolved_device=device,
        extra={"dummy": bool(args.dummy), "image_size": int(args.image_size)},
    )
    bbox_mapping = load_bbox_mapping(args.bbox_json) if args.bbox_json else {}
    output_dir = ensure_dir(args.output_dir)
    runtime = build_retrieval_runtime(
        memory_bank=args.memory_bank,
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
        uncertainty_threshold=args.uncertainty_threshold,
        uncertainty_scale=args.uncertainty_scale,
        policy_activation_threshold=args.policy_activation_threshold,
        residual_strength=args.residual_strength,
        allow_dummy_fallback=args.dummy,
    )

    rows = []
    diagnostics_rows = []
    input_images = collect_input_images(args.input_path)
    for index, image_path in enumerate(input_images, start=1):
        print(json.dumps({"progress": {"script": "run_retrieval_inference", "index": index, "total": len(input_images), "device": device}}, ensure_ascii=True), flush=True)
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
        retrieval = resolve_retrieval(
            runtime,
            query_feature,
            top_k_positive=args.top_k_positive,
            top_k_negative=args.top_k_negative,
            retrieval_mode=args.retrieval_mode,
            query_source=query_source,
            prefer_cross_domain_positive=True,
            sample_metadata=sample_metadata,
        )
        outputs, _, adapter_aux, _ = run_retrieval_forward(
            runtime,
            images=images,
            boxes=scaled_box,
            text_prompt=text_prompt,
            query_feature=query_feature,
            retrieval=retrieval,
            retrieval_mode=args.retrieval_mode,
            baseline_mask_logits=baseline_outputs.get("mask_logits"),
        )

        prob = torch.sigmoid(outputs["mask_logits"])[:, :1]
        resized_prob = F.interpolate(prob, size=(original_image.height, original_image.width), mode="bilinear", align_corners=False)[0, 0]
        binary_mask = (resized_prob > 0.5).detach().cpu().numpy().astype(np.uint8)
        mask_path = output_dir / f"{image_path.stem}_mask.png"
        overlay_path = output_dir / f"{image_path.stem}_overlay.png"
        preview_path = output_dir / f"{image_path.stem}_retrieved_preview.png"
        Image.fromarray(binary_mask * 255).save(mask_path)
        _overlay_image(original_image, binary_mask.astype(np.float32)).save(overlay_path)

        diagnostics = build_retrieval_diagnostics(
            image_id=image_path.stem,
            retrieval=retrieval,
            adapter_aux=adapter_aux,
            outputs=outputs,
            batch_index=0,
            sample_metadata={"image_id": image_path.stem, "image_path": str(image_path)},
        )
        diagnostics_rows.append(diagnostics)
        _save_preview(
            diagnostics["top_k_retrieved_exemplars"]["positive"],
            diagnostics["top_k_retrieved_exemplars"]["negative"],
            preview_path,
        )
        rows.append(
            {
                "image_id": image_path.stem,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "overlay_path": str(overlay_path),
                "retrieved_preview_path": str(preview_path),
                "similarity_score": diagnostics["similarity_score"],
                "retrieval_backend": runtime.retrieval_backend,
                "similarity_weighting": args.similarity_weighting,
                "similarity_temperature": args.similarity_temperature,
                "retrieval_policy": args.retrieval_policy,
            }
        )

    write_retrieval_diagnostics(output_dir / "retrieval_diagnostics.jsonl", diagnostics_rows)
    diagnostics_summary = summarize_retrieval_diagnostics(diagnostics_rows)
    (output_dir / "inference_results.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "count": len(rows),
                "bank_context": {
                    "resolved_path": str(runtime.bank_context.resolved_path),
                    "source": runtime.bank_context.source,
                    "cache_root": None if runtime.bank_context.cache_root is None else str(runtime.bank_context.cache_root),
                    "stats": runtime.bank_context.stats,
                },
                "retrieval_backend": runtime.retrieval_backend,
                "retrieval_diagnostics_summary": diagnostics_summary,
                "artifacts": {
                    "results": str(output_dir / "inference_results.jsonl"),
                    "retrieval_diagnostics": str(output_dir / "retrieval_diagnostics.jsonl"),
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