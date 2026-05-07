"""Train exemplar prompt adapter and prototype fusion for MedEx-SAM3."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import warnings

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from MedicalSAM3.adapters.exemplar_prompt_adapter import ExemplarPromptAdapter
from MedicalSAM3.exemplar.losses import MedExLossComposer
from MedicalSAM3.exemplar.memory_bank import ExemplarItem, ExemplarMemoryBank
from MedicalSAM3.exemplar.prototype_builder import PrototypeBuilder
from MedicalSAM3.sam3_official.build_model import build_official_sam3_image_model, freeze_model
from MedicalSAM3.sam3_official.tensor_forward import Sam3TensorForwardWrapper
from MedicalSAM3.scripts.common import (
    MedExSam3SegmentationModel,
    SplitSegmentationDataset,
    collate_batch,
    compute_segmentation_metrics,
    dump_config,
    ensure_dir,
    load_config,
    read_records,
    seed_everything,
)


def _prototype_summary(proto: torch.Tensor | None) -> torch.Tensor | None:
    if proto is None:
        return None
    if proto.dim() == 2:
        return proto.mean(dim=0, keepdim=True)
    return proto.mean(dim=1)


def _infer_hidden_dim(
    base_model: torch.nn.Module,
    wrapper: Sam3TensorForwardWrapper,
    device: str,
    image_size: int,
) -> int:
    hidden_dim = getattr(base_model, "hidden_dim", None)
    if hidden_dim is None:
        hidden_dim = getattr(base_model, "_medex_hidden_dim", None)
    if hidden_dim is None:
        hidden_dim = getattr(base_model, "embed_dim", None)
    if hidden_dim is not None:
        return int(hidden_dim)

    with torch.no_grad():
        outputs = wrapper(
            images=torch.rand(1, 3, image_size, image_size, device=device),
            boxes=torch.tensor([[image_size * 0.2, image_size * 0.2, image_size * 0.8, image_size * 0.8]], device=device),
            text_prompt=["polyp"],
        )
    for key in ["prompt_embeddings", "detector_queries", "image_embeddings"]:
        value = outputs.get(key)
        if not isinstance(value, torch.Tensor):
            continue
        if value.dim() >= 3:
            return int(value.shape[-1]) if key != "image_embeddings" else int(value.shape[1])
        if value.dim() == 2:
            return int(value.shape[-1])

    raise RuntimeError("Unable to infer SAM3 hidden dim from model attributes or wrapper outputs.")


def _select_top_items(builder: PrototypeBuilder, query: torch.Tensor, items: list[ExemplarItem], top_k: int) -> list[tuple[float, ExemplarItem, torch.Tensor]]:
    scored: list[tuple[float, ExemplarItem, torch.Tensor]] = []
    for item in items:
        embedding = builder._load_embedding(item).float().to(query.device)  # noqa: SLF001
        score = builder._item_score(query, embedding, item)  # noqa: SLF001
        scored.append((score, item, torch.nn.functional.normalize(embedding, dim=0)))
    scored.sort(key=lambda entry: entry[0], reverse=True)
    return scored[: min(top_k, len(scored))]


def _build_type_prototype(
    builder: PrototypeBuilder,
    query: torch.Tensor,
    items: list[ExemplarItem],
    top_k: int,
    mode: str,
) -> dict[str, object]:
    selected = _select_top_items(builder, query, items, top_k)
    if not selected:
        return {"prototype": None, "selected_item_ids": [], "weights": [], "variance": None}

    embeddings = torch.stack([entry[2] for entry in selected], dim=0)
    score_tensor = torch.tensor([entry[0] for entry in selected], dtype=torch.float32, device=query.device)

    if mode == "mean":
        prototype = builder.build_mean_prototype(embeddings)
        weights = torch.full((embeddings.shape[0],), 1.0 / embeddings.shape[0], device=query.device)
    elif mode == "weighted_mean":
        prototype, weights = builder.build_weighted_prototype(embeddings, score_tensor)
    elif mode == "attention_fusion":
        prototype, weights = builder.build_attention_fused_prototype(query, embeddings)
    elif mode == "clustered_subprototypes":
        prototype = builder.build_clustered_subprototypes(embeddings, n_clusters=min(top_k, embeddings.shape[0]))
        weights = torch.softmax(score_tensor, dim=0)
    else:
        raise ValueError(f"Unsupported prototype mode: {mode}")

    variance = builder.compute_prototype_variance(embeddings, prototype)
    return {
        "prototype": prototype,
        "selected_item_ids": [entry[1].item_id for entry in selected],
        "weights": weights.detach().cpu().tolist(),
        "variance": float(variance.item()),
    }


def _write_preflight_report(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Train exemplar prompt adapter for MedEx-SAM3.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--memory-bank", default="MedicalSAM3/outputs/medex_sam3/exemplar_bank")
    parser.add_argument("--prototype-mode", choices=["mean", "weighted_mean", "attention_fusion", "clustered_subprototypes"], default="weighted_mean")
    parser.add_argument("--top-k-positive", type=int, default=3)
    parser.add_argument("--top-k-negative", type=int, default=1)
    parser.add_argument("--top-k-boundary", type=int, default=1)
    parser.add_argument("--enable-negative-suppression", action="store_true")
    parser.add_argument("--enable-consistency-loss", action="store_true")
    parser.add_argument("--enable-contrastive-loss", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split-file", default="MedicalSAM3/outputs/medex_sam3/splits/fold_0/train_ids.txt")
    parser.add_argument("--output-dir", default="MedicalSAM3/outputs/medex_sam3/exemplar_prompt")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--precision", default="fp32")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dummy", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(int(config.get("seed", 42)))
    output_dir = ensure_dir(args.output_dir)
    preflight_report_path = output_dir / "preflight_report.json"
    variance_log = output_dir / "prototype_variance_log.jsonl"
    selected_log = output_dir / "selected_exemplars.jsonl"
    variance_log.write_text("", encoding="utf-8")
    selected_log.write_text("", encoding="utf-8")

    records = read_records(args.split_file)
    if args.dummy and not records:
        records = [{"image_path": "", "mask_path": "", "dataset_name": "dummy", "image_id": f"train_{i}"} for i in range(4)]
    if args.batch_size != 1:
        raise ValueError("train_exemplar_prompt.py currently expects --batch-size 1.")

    memory_bank_path = Path(args.memory_bank)
    bank_exists = memory_bank_path.exists()
    bank = ExemplarMemoryBank.load(args.memory_bank)
    positive_items = bank.get_items(type="positive", human_verified=True)
    negative_items = bank.get_items(type="negative", human_verified=True)
    boundary_items = bank.get_items(type="boundary", human_verified=True)
    memory_clean = bank.check_no_external_leakage(["PolypGen"])

    enable_negative_suppression = args.enable_negative_suppression
    enable_boundary = args.top_k_boundary > 0
    warnings_list: list[str] = []
    if enable_negative_suppression and not negative_items:
        enable_negative_suppression = False
        warnings_list.append("negative suppression disabled because there are no human-verified negative items")
    if enable_boundary and not boundary_items:
        enable_boundary = False
        warnings_list.append("boundary prototype disabled because there are no human-verified boundary items")

    preflight: dict[str, object] = {
        "memory_bank_exists": bank_exists,
        "trainable_items": len(bank.trainable_items),
        "positive_human_verified_items": len(positive_items),
        "negative_human_verified_items": len(negative_items),
        "boundary_human_verified_items": len(boundary_items),
        "memory_bank_no_polypgen_leakage": memory_clean,
        "enable_negative_suppression": enable_negative_suppression,
        "enable_boundary": enable_boundary,
        "prototype_mode": args.prototype_mode,
        "ready_for_training": False,
        "approximate_negative_suppression": enable_negative_suppression,
        "warnings": warnings_list,
        "error": None,
    }

    try:
        if not bank_exists:
            raise FileNotFoundError("Memory bank path does not exist.")
        if not bank.trainable_items:
            raise RuntimeError("Human-verified memory bank is empty. Run update_memory_from_review.py first.")
        if not positive_items:
            raise RuntimeError("No human-verified positive exemplars available for training.")
        if not memory_clean:
            raise RuntimeError("PolypGen leakage detected in memory bank.")
        if not records:
            raise FileNotFoundError("Split file is empty for exemplar prompt training.")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        base_model = build_official_sam3_image_model(
            args.checkpoint,
            device=device,
            dtype=args.precision,
            compile_model=False,
            allow_dummy_fallback=args.dummy,
        )
        freeze_model(base_model)
        wrapper = Sam3TensorForwardWrapper(model=base_model, device=device, dtype=args.precision)
        hidden_dim = _infer_hidden_dim(base_model, wrapper, device=device, image_size=args.image_size)
        model = MedExSam3SegmentationModel(
            wrapper=wrapper,
            enable_medical_adapter=True,
            enable_boundary_adapter=True,
            embed_dim=hidden_dim,
        ).to(device)
        prompt_adapter = ExemplarPromptAdapter(hidden_dim).to(device)

        if prompt_adapter.positive_proj.proj[0].in_features != hidden_dim:
            raise RuntimeError("ExemplarPromptAdapter dim does not match SAM3 hidden dim.")

        preflight["hidden_dim"] = hidden_dim
        preflight["ready_for_training"] = True
    except Exception as exc:
        preflight["error"] = str(exc)
        _write_preflight_report(preflight_report_path, preflight)
        if args.preflight_only:
            print(json.dumps(preflight, indent=2))
            return 0
        raise

    _write_preflight_report(preflight_report_path, preflight)
    if args.preflight_only:
        print(json.dumps(preflight, indent=2))
        return 0

    builder = PrototypeBuilder()
    criterion = MedExLossComposer()
    optimizer = AdamW(
        list(prompt_adapter.parameters()) + [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config.get("lr", 1e-4)),
    )
    loader = DataLoader(
        SplitSegmentationDataset(records, args.image_size),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )

    approximate_negative_suppression = enable_negative_suppression
    for epoch in range(args.epochs):
        model.train()
        for batch in loader:
            images = batch["images"].to(device)
            masks = batch["masks"].to(device)
            boxes = batch["boxes"].to(device)
            warmup_outputs = model(images=images, boxes=boxes, text_prompt=batch["text_prompt"], gt_mask=masks)
            query_embedding = warmup_outputs["query_embedding"].detach()[0]

            positive = _build_type_prototype(builder, query_embedding, positive_items, args.top_k_positive, args.prototype_mode)
            negative = _build_type_prototype(builder, query_embedding, negative_items, args.top_k_negative, args.prototype_mode)
            boundary = _build_type_prototype(builder, query_embedding, boundary_items, args.top_k_boundary, args.prototype_mode) if enable_boundary else {"prototype": None, "selected_item_ids": [], "weights": [], "variance": None}
            if positive["prototype"] is None:
                raise RuntimeError("No positive prototypes available for exemplar training.")

            positive_proto = positive["prototype"].unsqueeze(0) if positive["prototype"].dim() == 1 else positive["prototype"].unsqueeze(0)
            negative_proto = None if negative["prototype"] is None else negative["prototype"].unsqueeze(0) if negative["prototype"].dim() == 1 else negative["prototype"].unsqueeze(0)
            boundary_proto = None if boundary["prototype"] is None else boundary["prototype"].unsqueeze(0) if boundary["prototype"].dim() == 1 else boundary["prototype"].unsqueeze(0)

            prompt_tokens, prompt_aux = prompt_adapter(
                positive_proto=positive_proto,
                negative_proto=negative_proto,
                boundary_proto=boundary_proto,
                query_feat=warmup_outputs["query_embedding"],
            )
            outputs = model(
                images=images,
                boxes=boxes,
                text_prompt=batch["text_prompt"],
                exemplar_prompt_tokens=prompt_tokens,
                gt_mask=masks,
            )

            consistency_pair = None
            if args.enable_consistency_loss:
                alt_outputs = model(
                    images=images,
                    boxes=boxes,
                    text_prompt=batch["text_prompt"],
                    exemplar_prompt_tokens=prompt_tokens.flip(1),
                    gt_mask=masks,
                )
                consistency_pair = (outputs["mask_logits"], alt_outputs["mask_logits"])

            optimizer.zero_grad(set_to_none=True)
            loss, _ = criterion(
                outputs["mask_logits"],
                masks,
                anchor_embedding=outputs["query_embedding"] if args.enable_contrastive_loss and negative_proto is not None else None,
                positive_embedding=_prototype_summary(positive_proto) if args.enable_contrastive_loss and negative_proto is not None else None,
                negative_embeddings=negative_proto if args.enable_contrastive_loss and negative_proto is not None else None,
                negative_prompt_mask_logits=outputs["mask_logits"] * prompt_aux["suppression_gate"].view(-1, 1, 1, 1)
                if approximate_negative_suppression and negative_proto is not None
                else None,
                consistency_pair=consistency_pair,
            )
            loss.backward()
            optimizer.step()

            with variance_log.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "prototype_mode": args.prototype_mode,
                            "positive_variance": positive["variance"],
                            "negative_variance": negative["variance"],
                            "boundary_variance": boundary["variance"],
                        }
                    )
                    + "\n"
                )
            with selected_log.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "prototype_mode": args.prototype_mode,
                            "positive_ids": positive["selected_item_ids"],
                            "negative_ids": negative["selected_item_ids"],
                            "boundary_ids": boundary["selected_item_ids"],
                            "positive_weights": positive["weights"],
                            "negative_weights": negative["weights"],
                            "boundary_weights": boundary["weights"],
                            "approximate_negative_suppression": approximate_negative_suppression,
                        }
                    )
                    + "\n"
                )

    metrics = compute_segmentation_metrics(outputs["mask_logits"].detach(), masks.detach())
    metrics["approximate_negative_suppression"] = approximate_negative_suppression
    (output_dir / "val_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save(prompt_adapter.state_dict(), output_dir / "prompt_adapter.pt")
    dump_config(
        output_dir / "config_used.yaml",
        {
            **config,
            "prototype_mode": args.prototype_mode,
            "top_k_positive": args.top_k_positive,
            "top_k_negative": args.top_k_negative,
            "top_k_boundary": args.top_k_boundary,
            "dummy": args.dummy,
        },
    )
    print(json.dumps({"output_dir": str(output_dir), "metrics": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
