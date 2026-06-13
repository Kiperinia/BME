import argparse
from ultralytics import YOLO

def evaluate(weights="models/yolov8_polyp.pt", data="data/data.yaml"):
    # Load trained model
    model = YOLO(weights)

    # Evaluate
    metrics = model.val(data=data)
    print(f"mAP@50: {metrics.box.map50}")
    print(f"mAP@50-95: {metrics.box.map}")
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Polyp Detection Evaluation")
    parser.add_argument("--weights", type=str, default="models/yolov8_polyp.pt", help="Path to model weights")
    parser.add_argument("--data", type=str, default="data/data.yaml", help="Path to data config")
    
    args = parser.parse_args()
    evaluate(args.weights, args.data)
