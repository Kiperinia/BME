"""Train exemplar prompt adapter and prototype fusion for MedEx-SAM3."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from MedicalSAM3.adapters.exemplar_prompt_adapter import ExemplarPromptAdapter
from MedicalSAM3.exemplar.losses import MedExLossComposer
from MedicalSAM3.exemplar.memory_bank import ExemplarMemoryBank
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


def _build_type_prototype(builder: PrototypeBuilder, query: torch.Tensor, bank: ExemplarMemoryBank, exemplar_type: str, top_k: int):
    items = bank.get_items(type=exemplar_type, human_verified=True)
    return builder._build_single_type(query, items, top_k)  # noqa: SLF001


def _prototype_summary(proto: torch.Tensor | None) -> torch.Tensor | None:
    if proto is None:
        return None
    if proto.dim() == 2:
        return proto
    return proto.mean(dim=1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train exemplar prompt adapter for MedEx-SAM3.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--memory-bank", default="MedicalSAM3/outputs/medex_sam3/exemplar_bank")
    parser.add_argument("--prototype-mode", default="weighted_mean")
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
    parser.add_argument("--dummy", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(int(config.get("seed", 42)))
    records = read_records(args.split_file)
    if args.dummy and not records:
        records = [{"image_path": "", "mask_path": "", "dataset_name": "dummy", "image_id": f"train_{i}"} for i in range(4)]
    bank = ExemplarMemoryBank.load(args.memory_bank)
    if not bank.trainable_items:
        raise RuntimeError("Human-verified memory bank is empty. Run update_memory_from_review.py first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = build_official_sam3_image_model(args.checkpoint, device=device, dtype=args.precision, compile_model=False)
    freeze_model(base_model)
    wrapper = Sam3TensorForwardWrapper(model=base_model, device=device, dtype=args.precision)
    embed_dim = int(getattr(base_model, "embed_dim", 128))
    model = MedExSam3SegmentationModel(wrapper=wrapper, enable_medical_adapter=True, enable_boundary_adapter=True, embed_dim=embed_dim).to(device)
    prompt_adapter = ExemplarPromptAdapter(embed_dim).to(device)
    builder = PrototypeBuilder()
    criterion = MedExLossComposer()
    optimizer = AdamW(list(prompt_adapter.parameters()) + [parameter for parameter in model.parameters() if parameter.requires_grad], lr=float(config.get("lr", 1e-4)))

    loader = DataLoader(
        SplitSegmentationDataset(records, args.image_size),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    output_dir = ensure_dir(args.output_dir)
    variance_log = output_dir / "prototype_variance_log.jsonl"
    selected_log = output_dir / "selected_exemplars.jsonl"

    for epoch in range(args.epochs):
        model.train()
        for batch in loader:
            images = batch["images"].to(device)
            masks = batch["masks"].to(device)
            boxes = batch["boxes"].to(device)
            warmup_outputs = model(images=images, boxes=boxes, text_prompt=batch["text_prompt"], gt_mask=masks)
            query_embedding = warmup_outputs["query_embedding"].detach()[0]
            positive = _build_type_prototype(builder, query_embedding, bank, "positive", args.top_k_positive)
            negative = _build_type_prototype(builder, query_embedding, bank, "negative", args.top_k_negative)
            boundary = _build_type_prototype(builder, query_embedding, bank, "boundary", args.top_k_boundary)
            if positive["prototype"] is None:
                raise RuntimeError("No positive prototypes available for exemplar training.")

            positive_proto = positive["prototype"].unsqueeze(0) if positive["prototype"].dim() == 1 else positive["prototype"].unsqueeze(0)
            negative_proto = None if negative["prototype"] is None else negative["prototype"].unsqueeze(0)
            boundary_proto = None if boundary["prototype"] is None else boundary["prototype"].unsqueeze(0)
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
            loss, aux = criterion(
                outputs["mask_logits"],
                masks,
                anchor_embedding=outputs["query_embedding"] if args.enable_contrastive_loss and negative_proto is not None else None,
                positive_embedding=_prototype_summary(positive_proto) if args.enable_contrastive_loss and negative_proto is not None else None,
                negative_embeddings=negative_proto if args.enable_contrastive_loss and negative_proto is not None else None,
                negative_prompt_mask_logits=outputs["mask_logits"] * prompt_aux["suppression_gate"].view(-1, 1, 1, 1)
                if args.enable_negative_suppression and negative_proto is not None
                else None,
                consistency_pair=consistency_pair,
            )
            loss.backward()
            optimizer.step()

            with variance_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "epoch": epoch,
                    "positive_variance": positive["variance"],
                    "negative_variance": negative["variance"],
                    "boundary_variance": boundary["variance"],
                }) + "\n")
            with selected_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "epoch": epoch,
                    "positive_ids": positive["selected_item_ids"],
                    "negative_ids": negative["selected_item_ids"],
                    "boundary_ids": boundary["selected_item_ids"],
                    "positive_weights": positive["weights"],
                    "negative_weights": negative["weights"],
                    "boundary_weights": boundary["weights"],
                }) + "\n")

    metrics = compute_segmentation_metrics(outputs["mask_logits"].detach(), masks.detach())
    (output_dir / "val_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save(prompt_adapter.state_dict(), output_dir / "prompt_adapter.pt")
    dump_config(output_dir / "config_used.yaml", {
        **config,
        "prototype_mode": args.prototype_mode,
        "top_k_positive": args.top_k_positive,
        "top_k_negative": args.top_k_negative,
        "top_k_boundary": args.top_k_boundary,
        "dummy": args.dummy,
    })
    print(json.dumps({"output_dir": str(output_dir), "metrics": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
