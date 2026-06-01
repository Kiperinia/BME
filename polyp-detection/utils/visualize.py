import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os

def visualize_results(image_path, model_path="models/yolov8_polyp.pt"):
    model = YOLO(model_path)
    results = model(image_path)
    
    # Plot results
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 10))
        plt.imshow(im_rgb)
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    # Example usage
    sample_img = "data/images/val/sample.jpg"
    if os.path.exists(sample_img):
        visualize_results(sample_img)
    else:
        print(f"Sample image {sample_img} not found for visualization demo.")
