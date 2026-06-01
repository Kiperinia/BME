import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def train(args):
    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        plots=True,
        patience=args.patience,
    )

    save_dir = Path(getattr(model.trainer, "save_dir", Path(args.project) / args.name))
    best_weights = save_dir / "weights" / "best.pt"
    if best_weights.exists() and args.export_weights:
        export_path = Path(args.export_weights)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weights, export_path)
        print(f"Copied best weights to {export_path}")

    metrics = model.val(data=args.data, device=args.device)
    print("Evaluation metrics:", metrics)
    return results, metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 for polyp detection.")
    parser.add_argument("--data", default="data/data.yaml", help="YOLO dataset config.")
    parser.add_argument("--model", default="yolov8n.pt", help="Base model or checkpoint.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--device", default="cpu", help="Training device, e.g. cpu, 0, 0,1.")
    parser.add_argument("--workers", type=int, default=0, help="Data loader workers.")
    parser.add_argument("--project", default="runs/train", help="Output project directory.")
    parser.add_argument("--name", default="polyp_detection", help="Run name.")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience.")
    parser.add_argument(
        "--export-weights",
        default="models/yolov8_polyp.pt",
        help="Copy best.pt here after training. Use an empty string to skip.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
