import argparse
import json
import shutil
from pathlib import Path


DEFAULT_SEQUENCES = ["seq8", "seq10", "seq15", "seq16", "seq23"]


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Extract selected PolypGen-SAM3 sequences for MedicalSAM3 inference."
    )
    parser.add_argument(
        "--input-dir",
        default=str(repo_root / "MedicalSAM3" / "data" / "PolypGen-SAM3"),
        help="Source PolypGen-SAM3 directory containing images, masks, and prompts.json.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(repo_root / "MedicalSAM3" / "data" / "PolypGen-SAM3-seq8_10_15_16_23"),
        help="Output directory for the extracted subset.",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=DEFAULT_SEQUENCES,
        help="Sequence prefixes to extract.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the output directory before extracting.",
    )
    return parser


def should_keep(file_name: str, sequences: list[str]) -> bool:
    return any(file_name.startswith(seq) for seq in sequences)


def copy_matching_files(src_dir: Path, dst_dir: Path, sequences: list[str]) -> list[str]:
    copied_names = []
    dst_dir.mkdir(parents=True, exist_ok=True)
    for file_path in sorted(src_dir.iterdir()):
        if not file_path.is_file():
            continue
        if not should_keep(file_path.name, sequences):
            continue
        shutil.copy2(file_path, dst_dir / file_path.name)
        copied_names.append(file_path.name)
    return copied_names


def main() -> int:
    args = build_parser().parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    images_dir = input_dir / "images"
    masks_dir = input_dir / "masks"
    prompts_path = input_dir / "prompts.json"

    if not images_dir.is_dir() or not masks_dir.is_dir() or not prompts_path.is_file():
        raise FileNotFoundError("Input directory must contain images/, masks/, and prompts.json")

    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)

    output_images = output_dir / "images"
    output_masks = output_dir / "masks"

    copied_images = copy_matching_files(images_dir, output_images, args.sequences)
    copied_masks = copy_matching_files(masks_dir, output_masks, args.sequences)

    prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
    filtered_prompts = {
        key: value
        for key, value in prompts.items()
        if should_keep(key, args.sequences)
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prompts.json").write_text(
        json.dumps(filtered_prompts, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    copied_image_set = set(copied_images)
    copied_mask_set = set(copied_masks)
    missing_masks = sorted(copied_image_set - copied_mask_set)
    missing_images = sorted(copied_mask_set - copied_image_set)

    summary = {
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "sequences": args.sequences,
        "num_images": len(copied_images),
        "num_masks": len(copied_masks),
        "num_prompts": len(filtered_prompts),
        "missing_masks": missing_masks,
        "missing_images": missing_images,
    }
    (output_dir / "extract_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())