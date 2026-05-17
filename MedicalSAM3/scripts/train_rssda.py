"""Train retrieval-conditioned spatial-semantic domain adaptation for MedEx-SAM3."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from MedicalSAM3.adapters import RetrievalSpatialSemanticAdapter
from MedicalSAM3.adapters.lora import LoRAConfig, apply_lora_to_model, load_lora_weights
from MedicalSAM3.exemplar.losses import CrossDomainConsistencyLoss, MedExLossComposer
from MedicalSAM3.exemplar_bank import RSSDABank
from MedicalSAM3.models.retrieval import PrototypeRetriever, SimilarityHeatmapBuilder
from MedicalSAM3.retrieval import load_retrieval_bank
from MedicalSAM3.sam3_official.build_model import build_official_sam3_image_model, freeze_model
from MedicalSAM3.sam3_official.tensor_forward import Sam3TensorForwardWrapper
from MedicalSAM3.scripts.common import (
    SplitSegmentationDataset,
    apply_config_overrides,
    collate_batch,
    ensure_dir,
    infer_source_domain,
    load_config,
    log_runtime_environment,
    read_records,
    resolve_runtime_device,
    resolve_feature_map,
    seed_everything,
)


def _resolve_hidden_dim(model: torch.nn.Module) -> int:
    return int(getattr(model, "hidden_dim", getattr(model, "_medex_hidden_dim", getattr(model, "embed_dim", 128))))


def _cross_domain_consistency_loss(
    criterion: CrossDomainConsistencyLoss,
    query_global: torch.Tensor,
    positive_prototype: torch.Tensor,
    retrieved_entries: list[list[object]],
    records: list[dict[str, object]],
) -> torch.Tensor:
    penalties = []
    for batch_index, entries in enumerate(retrieved_entries):
        record_source = infer_source_domain(
            dataset_name=str(records[batch_index].get("dataset_name", "")),
            image_id=str(records[batch_index].get("image_id", "")),
            image_path=str(records[batch_index].get("image_path", "")),
            mask_path=str(records[batch_index].get("mask_path", "")),
        )
        if any(str(entry.source_dataset) != record_source for entry in entries):
            penalties.append(
                criterion(
                    query_global[batch_index: batch_index + 1],
                    positive_prototype[batch_index: batch_index + 1],
                )
            )
    if not penalties:
        return query_global.new_tensor(0.0)
    return torch.stack(penalties).mean()


def _apply_retrieval_mode(retrieval: dict[str, object], mode: str) -> dict[str, object]:
    if mode not in {"joint", "semantic", "spatial", "positive-only", "negative-only", "positive-negative"}:
        raise ValueError(f"Unsupported retrieval mode: {mode}")
    if mode in {"joint", "semantic", "spatial", "positive-negative"}:
        return retrieval
    updated = dict(retrieval)
    if mode == "positive-only":
        positive_prototype = retrieval["positive_prototype"]
        updated["negative_features"] = torch.zeros_like(retrieval["negative_features"])
        updated["negative_weights"] = torch.zeros_like(retrieval["negative_weights"])
        updated["negative_score_tensor"] = torch.zeros_like(retrieval.get("negative_score_tensor", retrieval["negative_weights"]))
        updated["negative_prototype"] = torch.zeros_like(positive_prototype)
        updated["negative_entries"] = [[] for _ in retrieval["positive_entries"]]
        updated["negative_scores"] = [torch.zeros_like(score) for score in retrieval["positive_scores"]]
        return updated
    negative_prototype = retrieval["negative_prototype"]
    updated["positive_features"] = torch.zeros_like(retrieval["positive_features"])
    updated["positive_weights"] = torch.zeros_like(retrieval["positive_weights"])
    updated["positive_score_tensor"] = torch.zeros_like(retrieval.get("positive_score_tensor", retrieval["positive_weights"]))
    updated["positive_prototype"] = torch.zeros_like(negative_prototype)
    updated["positive_entries"] = [[] for _ in retrieval["negative_entries"]]
    updated["positive_scores"] = [torch.zeros_like(score) for score in retrieval["negative_scores"]]
    return updated


def _load_checkpoint_payload(path: Path, device: str) -> object:
    return torch.load(path, map_location=device, weights_only=False)


def _maybe_load_rssda_bundle(
    path: Path,
    device: str,
    adapter: RetrievalSpatialSemanticAdapter,
    retriever: PrototypeRetriever,
    similarity_builder: SimilarityHeatmapBuilder,
) -> bool:
    payload = _load_checkpoint_payload(path, device)
    if not isinstance(payload, dict):
        return False
    loaded = False
    adapter_state = payload.get("adapter")
    retriever_state = payload.get("retriever")
    similarity_state = payload.get("similarity_builder")
    if isinstance(adapter_state, dict):
        adapter.load_state_dict(adapter_state, strict=False)
        loaded = True
    if isinstance(retriever_state, dict):
        retriever.load_state_dict(retriever_state, strict=False)
        loaded = True
    if isinstance(similarity_state, dict):
        similarity_builder.load_state_dict(similarity_state, strict=False)
        loaded = True
    return loaded


def main() -> int:
    parser = argparse.ArgumentParser(description="Train RSS-DA for MedEx-SAM3.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--memory-bank", default="MedicalSAM3/banks/train_bank")
    parser.add_argument("--bank-purpose", default="train", choices=["train", "validation", "external-eval", "continual-adaptation"])
    parser.add_argument("--split-file", default="MedicalSAM3/outputs/medex_sam3/splits/fold_0/train_ids.txt")
    parser.add_argument("--output-dir", default="MedicalSAM3/outputs/medex_sam3/rssda")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--adapter-checkpoint", default=None)
    parser.add_argument("--lora-checkpoint", default=None)
    parser.add_argument("--lora-stage", default="stage_a")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--precision", default="fp32")
    parser.add_argument("--top-k-positive", type=int, default=1)
    parser.add_argument("--top-k-negative", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--progress-log-interval", type=int, default=10)
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
    parser.add_argument("--w-cross-domain", type=float, default=0.05)
    parser.add_argument("--prompt-corruption-prob", type=float, default=0.6)
    parser.add_argument("--bbox-perturb-ratio", type=float, default=0.05)
    parser.add_argument("--bbox-perturb-prob", type=float, default=0.5)
    parser.add_argument("--loose-bbox-ratio", type=float, default=0.2)
    parser.add_argument("--loose-bbox-prob", type=float, default=0.3)
    parser.add_argument("--bbox-dropout-prob", type=float, default=0.1)
    parser.add_argument("--prompt-removal-prob", type=float, default=0.05)
    parser.add_argument("--retrieval-mode", choices=["joint", "semantic", "spatial", "positive-only", "negative-only", "positive-negative"], default="joint")
    parser.add_argument("--prefer-cross-domain-positive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
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
    seed_everything(int(config.get("seed", 42)))
    output_dir = ensure_dir(args.output_dir)
    run_manifest_path = output_dir / "run_manifest.json"
    train_log_path = output_dir / "train_log.jsonl"
    run_manifest_path.write_text(json.dumps({**vars(args), "config": config}, indent=2), encoding="utf-8")
    train_log_path.write_text("", encoding="utf-8")
    records = read_records(args.split_file)
    if args.dummy and not records:
        records = [{"image_path": "", "mask_path": "", "dataset_name": "dummy", "image_id": f"rssda_{index}"} for index in range(4)]
    if not records:
        raise FileNotFoundError("No train records found for RSS-DA training.")

    device = resolve_runtime_device(args.device)
    log_runtime_environment(
        "train_rssda",
        requested_device=args.device,
        resolved_device=device,
        extra={"dummy": bool(args.dummy), "image_size": int(args.image_size)},
    )
    bank_context = load_retrieval_bank(
        args.memory_bank,
        purpose=args.bank_purpose,
        checkpoint=args.checkpoint,
        device=device,
        precision=args.precision,
        image_size=args.image_size,
        allow_dummy_fallback=args.dummy,
    )
    bank = bank_context.bank
    run_manifest_path.write_text(
        json.dumps(
            {
                **vars(args),
                "config": config,
                "bank_context": {
                    "resolved_path": str(bank_context.resolved_path),
                    "source": bank_context.source,
                    "cache_root": None if bank_context.cache_root is None else str(bank_context.cache_root),
                    "stats": bank_context.stats,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not bank.get_entries(polarity="positive", human_verified=True):
        raise RuntimeError("Train retrieval bank is empty. Populate MedicalSAM3/banks/train_bank/positive first.")

    base_model = build_official_sam3_image_model(
        args.checkpoint,
        device=device,
        dtype=args.precision,
        compile_model=False,
        allow_dummy_fallback=args.dummy,
    )
    if args.lora_checkpoint and Path(args.lora_checkpoint).exists():
        apply_lora_to_model(base_model, LoRAConfig(stage=args.lora_stage, min_replaced_modules=0))
        load_lora_weights(base_model, args.lora_checkpoint, strict=False)
    freeze_model(base_model)
    wrapper = Sam3TensorForwardWrapper(model=base_model, device=device, dtype=args.precision)
    hidden_dim = _resolve_hidden_dim(base_model)
    retriever = PrototypeRetriever(
        bank=bank,
        feature_dim=hidden_dim,
        top_k_positive=args.top_k_positive,
        top_k_negative=args.top_k_negative,
    ).to(device)
    similarity_builder = SimilarityHeatmapBuilder(lambda_negative=args.negative_lambda).to(device)
    adapter = RetrievalSpatialSemanticAdapter(
        dim=hidden_dim,
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
    ).to(device)
    if args.adapter_checkpoint and Path(args.adapter_checkpoint).exists():
        loaded_bundle = _maybe_load_rssda_bundle(Path(args.adapter_checkpoint), device, adapter, retriever, similarity_builder)
        if not loaded_bundle:
            adapter.load_state_dict(_load_checkpoint_payload(Path(args.adapter_checkpoint), device), strict=False)
    optimizer = AdamW(
        list(adapter.parameters()) + list(retriever.parameters()) + list(similarity_builder.parameters()),
        lr=float(config.get("lr", 1e-4)),
    )
    criterion = MedExLossComposer()
    cross_domain_criterion = CrossDomainConsistencyLoss()
    loader = DataLoader(
        SplitSegmentationDataset(
            records,
            image_size=args.image_size,
            prompt_corruption_prob=args.prompt_corruption_prob,
            box_jitter_ratio=args.bbox_perturb_ratio,
            box_jitter_prob=args.bbox_perturb_prob,
            loose_box_ratio=args.loose_bbox_ratio,
            loose_box_prob=args.loose_bbox_prob,
            box_dropout_prob=args.bbox_dropout_prob,
            prompt_removal_prob=args.prompt_removal_prob,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )

    history = []
    progress_log_interval = max(int(args.progress_log_interval), 1)
    total_steps_per_epoch = len(loader)
    effective_steps_per_epoch = total_steps_per_epoch if args.max_steps is None else min(total_steps_per_epoch, max(int(args.max_steps), 0))
    for epoch in range(args.epochs):
        adapter.train()
        for step, batch in enumerate(loader, start=1):
            if args.max_steps is not None and step > int(args.max_steps):
                break
            if step == 1 or step % progress_log_interval == 0 or step == effective_steps_per_epoch:
                print(json.dumps({"progress": {"script": "train_rssda", "epoch": epoch + 1, "step": step, "steps_per_epoch": effective_steps_per_epoch, "device": device, "dummy": bool(args.dummy)}}, ensure_ascii=True), flush=True)
            images = batch["images"].to(device)
            masks = batch["masks"].to(device)
            boxes = batch["boxes"].to(device)
            warmup = wrapper(images=images, boxes=boxes, text_prompt=batch["text_prompt"])
            query_feature = resolve_feature_map(warmup["image_embeddings"], images)
            query_source_datasets = [
                infer_source_domain(
                    dataset_name=str(record.get("dataset_name", "")),
                    image_id=str(record.get("image_id", "")),
                    image_path=str(record.get("image_path", "")),
                    mask_path=str(record.get("mask_path", "")),
                )
                for record in batch["records"]
            ]
            retrieval = _apply_retrieval_mode(
                retriever(
                    query_feature,
                    query_source_datasets=query_source_datasets,
                    prefer_cross_domain_positive=args.prefer_cross_domain_positive,
                ),
                args.retrieval_mode,
            )
            similarity = similarity_builder(
                query_feature,
                retrieval["positive_features"],
                retrieval["negative_features"],
                retrieval["positive_weights"],
                retrieval["negative_weights"],
            )
            _, retrieval_prior, adapter_aux = adapter(
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
                baseline_mask_logits=warmup.get("mask_logits"),
                positive_heatmap=similarity["positive_heatmap"],
                negative_heatmap=similarity["negative_heatmap"],
                mode=args.retrieval_mode,
            )
            outputs = wrapper(
                images=images,
                boxes=boxes,
                text_prompt=batch["text_prompt"],
                retrieval_prior=retrieval_prior,
            )

            has_negative = retrieval["negative_features"].shape[1] > 0 and retrieval["negative_features"].abs().sum().item() > 0
            loss, aux = criterion(
                mask_logits=outputs["mask_logits"],
                gt_mask=masks,
                anchor_embedding=retrieval["query_global"] if has_negative else None,
                positive_embedding=retrieval["positive_prototype"] if has_negative else None,
                negative_embeddings=retrieval["negative_features"] if has_negative else None,
                negative_prompt_mask_logits=adapter_aux["negative_prompt_mask_logits"],
            )
            cross_domain = _cross_domain_consistency_loss(
                cross_domain_criterion,
                retrieval["projected_query"],
                retrieval["positive_prototype"],
                retrieval["positive_entries"],
                batch["records"],
            )
            cross_domain_pair_ratio = 0.0
            if retrieval["positive_entries"]:
                positive_cross_domain = 0
                for record_source, entries in zip(query_source_datasets, retrieval["positive_entries"]):
                    if any(str(entry.source_dataset) != record_source for entry in entries):
                        positive_cross_domain += 1
                cross_domain_pair_ratio = positive_cross_domain / max(len(retrieval["positive_entries"]), 1)
            total_loss = loss + args.w_cross_domain * cross_domain

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            history.append(
                {
                    "epoch": epoch,
                    "step": step,
                    "steps_per_epoch": effective_steps_per_epoch,
                    "loss": float(total_loss.item()),
                    "segmentation": float(aux["total"].item()),
                    "cross_domain": float(cross_domain.item()),
                    "cross_domain_pair_ratio": float(cross_domain_pair_ratio),
                    "retrieval_mode": args.retrieval_mode,
                }
            )
            with train_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(history[-1]) + "\n")

            if step >= effective_steps_per_epoch:
                break

    torch.save(adapter.state_dict(), output_dir / "retrieval_adapter.pt")
    torch.save(retriever.state_dict(), output_dir / "retriever_projection.pt")
    torch.save(similarity_builder.state_dict(), output_dir / "similarity_fusion.pt")
    torch.save(
        {
            "adapter": adapter.state_dict(),
            "retriever": retriever.state_dict(),
            "similarity_builder": similarity_builder.state_dict(),
            "args": vars(args),
        },
        output_dir / "rssda_state.pt",
    )
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    if history:
        (output_dir / "train_summary.json").write_text(json.dumps(history[-1], indent=2), encoding="utf-8")
    (output_dir / "config_used.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(json.dumps({"epochs": args.epochs, "steps": len(history), "steps_per_epoch": effective_steps_per_epoch, "output_dir": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())