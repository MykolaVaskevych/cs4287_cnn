#!/usr/bin/env python3
"""Create summary.json and summary.txt for completed experiments"""

import json
from pathlib import Path
import polars as pl
from datetime import datetime

experiments = {
    "ppe_100_lr_0.001": {
        "experiment_id": "lr_low",
        "description": "Low learning rate",
        "params": {"epochs": 100, "lr0": 0.001, "batch": 16, "imgsz": 640, "dropout": 0.0, "mosaic": 1.0}
    },
    "ppe_100_lr_0.02": {
        "experiment_id": "lr_high",
        "description": "High learning rate",
        "params": {"epochs": 100, "lr0": 0.02, "batch": 16, "imgsz": 640, "dropout": 0.0, "mosaic": 1.0}
    },
    "ppe_100_batch_8": {
        "experiment_id": "batch_small",
        "description": "Smaller batch size",
        "params": {"epochs": 100, "lr0": 0.01, "batch": 8, "imgsz": 640, "dropout": 0.0, "mosaic": 1.0}
    },
    "ppe_100_no_mosaic": {
        "experiment_id": "no_augment",
        "description": "No mosaic augmentation",
        "params": {"epochs": 100, "lr0": 0.01, "batch": 16, "imgsz": 640, "dropout": 0.0, "mosaic": 0.0}
    },
    "ppe_100_dropout_0.2": {
        "experiment_id": "dropout",
        "description": "With dropout regularization",
        "params": {"epochs": 100, "lr0": 0.01, "batch": 16, "imgsz": 640, "dropout": 0.2, "mosaic": 1.0}
    }
}

runs_dir = Path("runs/saved")

for run_name, exp_info in experiments.items():
    run_dir = runs_dir / run_name

    if not (run_dir / "results.csv").exists():
        print(f"⏭ Skipping {run_name} - no results.csv")
        continue

    print(f"📊 Processing {run_name}...")

    # Read results
    results_csv = pl.read_csv(run_dir / "results.csv")
    final_metrics = results_csv.tail(1)

    # Extract metrics
    mAP50 = final_metrics["metrics/mAP50(B)"][0]
    mAP5095 = final_metrics["metrics/mAP50-95(B)"][0]
    precision = final_metrics["metrics/precision(B)"][0]
    recall = final_metrics["metrics/recall(B)"][0]
    duration_hours = final_metrics["time"][0] / 3600

    # Create summary dict
    summary = {
        "experiment_id": exp_info["experiment_id"],
        "run_name": run_name,
        "description": exp_info["description"],
        "parameters": exp_info["params"],
        "duration_hours": duration_hours,
        "final_metrics": {
            "mAP50": float(mAP50),
            "mAP50-95": float(mAP5095),
            "precision": float(precision),
            "recall": float(recall),
        },
        "training_completed": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # Save summary.json
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Create text summary
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    text_summary = f"""
RUN: {run_name}
EXPERIMENT: {exp_info['experiment_id']} - {exp_info['description']}
COMPLETED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
DURATION: {duration_hours:.2f} hours

HYPERPARAMETERS:
  Learning Rate (lr0): {exp_info['params']['lr0']}
  Epochs: {exp_info['params']['epochs']}
  Batch Size: {exp_info['params']['batch']}
  Image Size: {exp_info['params']['imgsz']}
  Dropout: {exp_info['params']['dropout']}
  Mosaic Augmentation: {exp_info['params']['mosaic']}

FINAL METRICS:
  mAP@0.5: {mAP50:.4f}
  mAP@0.5:0.95: {mAP5095:.4f}
  Precision: {precision:.4f}
  Recall: {recall:.4f}
  F1 Score: {f1_score:.4f}

STATUS: ✓ SUCCESS
"""

    with open(run_dir / "summary.txt", "w") as f:
        f.write(text_summary)

    print(f"  ✓ mAP50: {mAP50:.4f} | Duration: {duration_hours:.2f}h")

print("\n✅ All summaries created!")
