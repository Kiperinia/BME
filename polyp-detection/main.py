import argparse
from ultralytics import YOLO
import os

def run_inference(source, weights="models/yolov8_polyp.pt"):
    """
    Run inference on a source (image, directory, or video) using the specified weights.
    """
    if not os.path.exists(weights):
        print(f"Error: Model weights not found at {weights}")
        return

    # Load trained model
    model = YOLO(weights)

    # Run inference
    results = model.predict(source=source, save=True, conf=0.25)
    
    print(f"Inference complete. Results saved to 'runs/detect'.")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Polyp Detection Inference")
    parser.add_argument("--source", type=str, default="data/images/val", help="Path to input image/video or directory")
    parser.add_argument("--weights", type=str, default="models/yolov8_polyp.pt", help="Path to model weights")
    
    args = parser.parse_args()
    
    run_inference(args.source, args.weights)
