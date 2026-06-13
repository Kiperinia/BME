import argparse
import json
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def yolo_line(box, width, height):
    xmin = max(0.0, float(box["xmin"]))
    ymin = max(0.0, float(box["ymin"]))
    xmax = min(float(width), float(box["xmax"]))
    ymax = min(float(height), float(box["ymax"]))

    if xmax <= xmin or ymax <= ymin:
        return None

    x_center = ((xmin + xmax) / 2.0) / width
    y_center = ((ymin + ymax) / 2.0) / height
    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height
    return f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"


def clear_split_dirs(output_dir):
    for folder in [
        "images/train",
        "images/val",
        "images/test",
        "labels/train",
        "labels/val",
        "labels/test",
        "test_images",
    ]:
        path = output_dir / folder
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def split_items(items, train_ratio, val_ratio):
    train_end = int(len(items) * train_ratio)
    val_end = train_end + int(len(items) * val_ratio)
    return {
        "train": items[:train_end],
        "val": items[train_end:val_end],
        "test": items[val_end:],
    }


def prepare_dataset(raw_dir, output_dir, seed, train_ratio, val_ratio, test_samples, limit):
    images_dir = raw_dir / "images"
    bbox_file = raw_dir / "kavsir_bboxes.json"

    if not images_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")
    if not bbox_file.exists():
        raise FileNotFoundError(f"Bounding-box JSON not found: {bbox_file}")

    with bbox_file.open("r", encoding="utf-8") as f:
        bboxes = json.load(f)

    image_paths = sorted(
        p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    usable = [p for p in image_paths if p.stem in bboxes]
    if limit:
        usable = usable[:limit]

    rng = random.Random(seed)
    rng.shuffle(usable)

    clear_split_dirs(output_dir)
    splits = split_items(usable, train_ratio, val_ratio)

    summary = {}
    for split_name, split_images in splits.items():
        summary[split_name] = {"images": 0, "labels": 0, "boxes": 0}
        for image_path in split_images:
            item = bboxes[image_path.stem]
            width = int(item["width"])
            height = int(item["height"])
            label_lines = [
                line
                for line in (yolo_line(box, width, height) for box in item.get("bbox", []))
                if line is not None
            ]
            if not label_lines:
                continue

            shutil.copy2(image_path, output_dir / "images" / split_name / image_path.name)
            label_path = output_dir / "labels" / split_name / f"{image_path.stem}.txt"
            label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")

            summary[split_name]["images"] += 1
            summary[split_name]["labels"] += 1
            summary[split_name]["boxes"] += len(label_lines)

    test_source = splits["test"][:test_samples] or splits["val"][:test_samples]
    for image_path in test_source:
        shutil.copy2(image_path, output_dir / "test_images" / image_path.name)

    manifest = {
        "source": str(raw_dir),
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": 1.0 - train_ratio - val_ratio,
        "total_source_images": len(image_paths),
        "usable_images": len(usable),
        "test_sample_images": len(test_source),
        "splits": summary,
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Prepare Kvasir-SEG as a YOLO detection dataset.")
    parser.add_argument("--raw-dir", default="data/raw/Kvasir-SEG", help="Extracted Kvasir-SEG directory.")
    parser.add_argument("--output-dir", default="data", help="Project data directory.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic shuffle seed.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Training split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--test-samples", type=int, default=12, help="Images copied to data/test_images.")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit for quick experiments.")
    args = parser.parse_args()

    manifest = prepare_dataset(
        raw_dir=Path(args.raw_dir),
        output_dir=Path(args.output_dir),
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_samples=args.test_samples,
        limit=args.limit or None,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
