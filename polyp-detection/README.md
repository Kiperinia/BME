# Polyp Detection with YOLOv8

This project implements a real-time polyp detection system using the YOLOv8 (You Only Look Once) architecture. It is designed for clinical assistance in identifying polyps during colonoscopy procedures.

## Why This Code Works

### 1. YOLOv8 Architecture
We use YOLOv8 because it provides a state-of-the-art balance between speed and accuracy. In clinical settings, real-time detection (high FPS) is critical to assist clinicians without latency. YOLOv8's anchor-free detection and advanced augmentation pipelines make it highly robust to varying polyp shapes and lighting conditions.

### 2. Standardized Data Pipeline
The project follows the standard YOLO format:
- **Images**: Located in `data/images/`
- **Labels**: YOLO-format `.txt` files in `data/labels/` (normalized coordinates: `<class_id> <x_center> <y_center> <width> <height>`)
- **Configuration**: `data/data.yaml` maps these paths for the training engine.

### 3. Modular Design
Dividing the codebase into `training`, `utils`, and `main` ensures:
- **Scalability**: New preprocessing or visualization techniques can be added without modifying core training logic.
- **Reproducibility**: Training parameters are centralized in `training/train.py`.

## Project Structure

```text
polyp-detection/
├── data/               # Dataset storage and config
│   ├── images/         # Train/Val/Test images
│   ├── labels/         # YOLO format labels
│   └── data.yaml       # Dataset configuration
├── training/           # Training and evaluation logic
│   ├── train.py        # Model training script
│   └── evaluate.py     # Detailed validation metrics
├── utils/              # Helper scripts
│   ├── preprocess.py   # Data cleaning and resizing
│   ├── visualize.py    # Result plotting
│   └── metrics.py      # Custom clinical metrics (Dice, etc.)
├── models/             # Saved model weights (.pt)
├── main.py             # Entry point for inference
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## Recommended Datasets

To get started quickly, you can download these free, high-quality datasets for polyp detection:

1.  **Kvasir-SEG** (1,000 images):
    *   [Official Website](https://datasets.simula.no/kvasir-seg/)
    *   [Kaggle (YOLO Formatted)](https://www.kaggle.com/datasets/saharmagdy/kvasir-dataset-for-yolo) - **Recommended for easy start.**
2.  **CVC-ClinicDB** (612 images):
    *   [Grand-Challenge Link](https://polyp.grand-challenge.org/CVCClinicDB/)
3.  **Roboflow Universe**:
    *   Search for ["Polyp Detection"](https://universe.roboflow.com/search?q=polyp+detection) to find community-shared datasets already formatted for YOLOv8.

## Setup and Preparation

### 1. Adding Your Data
The `data/` directory is currently structured to receive your images and labels. You must organize them as follows:

- **Images**: Place `.jpg` or `.png` files in `data/images/train` and `data/images/val`.
- **Labels**: Place corresponding YOLO `.txt` files in `data/labels/train` and `data/labels/val`.

> [!IMPORTANT]
> A label file must have the exact same name as its image (e.g., `frame_001.jpg` and `frame_001.txt`).

**Label Format:**
Each line in the `.txt` file should follow: `<class_id> <x_center> <y_center> <width> <height>` (normalized between 0 and 1).

### 2. Getting the Model
The `models/` directory is where you store your trained weights (`yolov8_polyp.pt`). 

- **For Training**: When you run `training/train.py`, YOLOv8 will automatically download the base weights (`yolov8n.pt`) from Ultralytics if they are not found.
- **For Inference**: Once training is complete, move the best weights from `runs/train/polyp_detection/weights/best.pt` to `models/yolov8_polyp.pt`.

## How To Run

### 1. Installation
```powershell
pip install -r requirements.txt
```

### 2. Training
Ensure your images and labels are in place, then run:
```powershell
python training/train.py
```

For a quick CPU smoke run:
```powershell
.\.venv\Scripts\python.exe training\train.py --epochs 1 --imgsz 320 --batch 4 --workers 0 --device cpu --name polyp_detection_smoke
```

### 3. Inference / Detection
Once you have a model in `models/yolov8_polyp.pt`, run:
```powershell
python main.py --source data/images/val
```

### 4. Dataset Annotation Agent
The full BME project includes an agent that checks whether images have YOLO bbox labels and automatically labels missing or invalid files with the trained detector.

From the BME project root:
```powershell
.\polyp-detection\.venv\Scripts\python.exe agent\run_annotation_agent.py --image-root polyp-detection\data --dataset-root polyp-detection\data
```

Use `--dry-run` to audit and predict without writing labels:
```powershell
.\polyp-detection\.venv\Scripts\python.exe agent\run_annotation_agent.py --image-root polyp-detection\data\test_images --dataset-root polyp-detection\data --dry-run
```

### 5. Hybrid Local + Gemini Diagnosis
The BME agent layer can combine local YOLO detection with Gemini image reasoning. Set `GEMINI_API_KEY` in the BME project root `.env` file, then run:

```powershell
.\polyp-detection\.venv\Scripts\python.exe agent\run_hybrid_diagnosis_agent.py --image polyp-detection\data\test_images\cju1ffnjn6ctm08015perkg37.jpg --output polyp-detection\runs\hybrid_diagnosis_sample.json
```
