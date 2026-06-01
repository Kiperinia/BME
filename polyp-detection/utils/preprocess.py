import cv2
import numpy as np

def preprocess_image(image_path, target_size=(640, 640)):
    """
    Resize and normalize image for YOLOv8.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    return img

def augment_data(image):
    """
    Apply simple horizontal flip augmentation.
    """
    return cv2.flip(image, 1)
