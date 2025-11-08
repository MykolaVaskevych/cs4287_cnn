# CS4287-CNN: Construction Safety Equipment Detection

A YOLOv8-based deep learning model for detecting Personal Protective Equipment (PPE) violations on construction sites.

## Overview

This project fine-tunes a YOLOv8 nano model to identify:
- Safety equipment: Hardhats, Masks, Safety Vests
- PPE violations: Workers without proper protection
- Construction site elements: Personnel, Machinery, Vehicles, Safety Cones

## Features

- **Automated GPU Detection**: Automatically detects and uses available NVIDIA GPU or falls back to CPU
- **Interactive Training**: Manual training trigger via button - no automatic execution
- **Comprehensive Validation**: Pre-training checks for dataset existence and structure
- **Results Analysis**: Confusion matrices, training metrics, and model comparison visualizations
- **Production Ready**: Reproducible results with fixed random seeds

## Requirements

### Hardware
- **GPU (Recommended)**: NVIDIA GPU with 8GB+ VRAM (If Available)
- **CPU**: 16GB+ RAM (training will be significantly slower)
- **Disk Space**: 5GB for dataset + 500MB for models

### Software
- Python 3.13+
- uv package manager
- CUDA toolkit (for GPU training)

## Setup

### 1. Install UV (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install Dependencies

```bash
# Sync all dependencies from pyproject.toml
uv sync
```

### 3. Download Dataset

Extract the [Construction Site Safety](https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow) dataset to:
```
data/archive/css-data/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### 4. Run Notebook

```bash
# Start marimo notebook in edit mode
uv run marimo edit main.py

# Or run in read-only mode
uv run marimo run main.py
```

## Usage Workflow

1. **Dataset Exploration**
   - Run all cells sequentially to validate dataset paths
   - Review class distribution and data quality
   - Examine sample images with annotations

2. **Baseline Evaluation**
   - Test pre-trained COCO model on test set
   - Establishes performance baseline

3. **Training** (Manual)
   - Adjust training parameters in constants cell if needed:
     - `DEVICE`: Auto-detected (0=GPU, 'cpu'=CPU)
     - `EPOCHS`: 50 (reduce to 10 for quick testing)
     - `BATCH_SIZE`: 16 (reduce to 4-8 for low VRAM)
   - Click "Train Model" button
   - Wait for training to complete (30-60 min on GPU)

4. **Results Analysis**
   - Review training metrics and confusion matrix
   - Compare pre-trained vs fine-tuned performance
   - Analyze model predictions on test set

## Configuration

All parameters are centralized in the constants cell:

```python
DEVICE = 0  # Auto-detected (0=GPU, 'cpu'=CPU)
EPOCHS = 50  # Reduce for quick testing
IMAGE_SIZE = 640  # YOLO standard
BATCH_SIZE = 16  # Adjust based on VRAM
CONFIDENCE_THRESHOLD = 0.25  # Detection confidence
RANDOM_SEED = 42  # Reproducibility
```

## Directory Structure

```
cs4287_cnn/
├── main.py                 # Marimo notebook
├── pyproject.toml          # Dependencies
├── CLAUDE.md               # Marimo assistant rules
├── README.md               # This file
├── data/
│   └── archive/
│       └── css-data/       # Dataset (not in git)
└── runs/                   # Training outputs (created during training)
    └── train/
        └── ppe_detection/
            ├── weights/
            │   └── best.pt # Best model checkpoint
            ├── results.png
            └── ...
```

## Model Usage

After training, use the model for inference:

```python
from ultralytics import YOLO

# Load trained model
model = YOLO("runs/train/ppe_detection/weights/best.pt")

# Run inference
results = model.predict(
    source="path/to/image.jpg",
    conf=0.25,
    save=True
)

# Access detections
for result in results:
    boxes = result.boxes
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"Class: {cls}, Confidence: {conf:.2f}")
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `BATCH_SIZE` in constants cell (try 8, then 4)
- Reduce `IMAGE_SIZE` to 416
- Close other GPU applications

### Slow Training
- Verify GPU is detected (check console output)
- Reduce `EPOCHS` for faster iteration
- Ensure CUDA drivers are installed

### Poor Performance
- Check class distribution for imbalance
- Increase `EPOCHS` (try 100)
- Try larger models (yolov8s.pt, yolov8m.pt)

### Import Errors
```bash
# Resync dependencies
uv sync

# Add specific package
uv add package-name

# Check Python version
python --version  # Should be 3.13+
```

## Performance Expectations

| Metric | Target | Interpretation |
|--------|--------|----------------|
| mAP50 | >0.7 | Excellent |
| mAP50 | >0.5 | Good |
| mAP50 | <0.5 | Needs improvement |

## Dataset Classes

| ID | Class | Color | Notes |
|----|-------|-------|-------|
| 0 | Hardhat | Green | Compliance |
| 1 | Mask | Cyan | Compliance |
| 2 | NO-Hardhat | Red | Violation |
| 3 | NO-Mask | Red | Violation |
| 4 | NO-Safety Vest | Red | Violation |
| 5 | Person | Magenta | Worker |
| 6 | Safety Cone | Orange | Equipment |
| 7 | Safety Vest | Green | Compliance |
| 8 | machinery | Gray | Equipment |
| 9 | vehicle | Blue | Equipment |

## Development

### Adding Dependencies

```bash
# Add runtime dependency
uv add package-name

# Add development dependency
uv add --dev package-name

# Add from git
uv add git+https://github.com/user/repo

# Sync environment after changes
uv sync
```

### Code Quality

```bash
# Validate notebook
uv run marimo check main.py

# Auto-fix issues
uv run marimo check --fix main.py
```

## Authors

- MYKOLA VASKEVYCH (22372199)
- OLIVER FITZGERALD (22365958)

## License

Academic project for CS4287 Neural Computing course.

## Acknowledgments

- YOLOv8 by Ultralytics
- Construction Site Safety dataset
- Marimo interactive notebook framework
