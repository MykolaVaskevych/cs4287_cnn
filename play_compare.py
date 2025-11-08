import marimo

__generated_with = "0.17.2"
app = marimo.App(width="full", auto_download=["ipynb"])


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from pathlib import Path
    import seaborn as sns
    from scipy import stats
    import yaml
    return Path, mo, mpimg, np, pl, plt, stats, yaml


@app.cell
def _(Path):
    # Define paths to training results
    BASE_DIR = Path.cwd()
    RUNS_DIR = BASE_DIR / "runs" / "saved"

    # Training run directories
    RUN_100_DIR = RUNS_DIR / "ppe_100"
    RUN_500_DIR = RUNS_DIR / "ppe_500"

    # Individual model checkpoints
    MODEL_10 = RUNS_DIR / "best_10.pt"
    MODEL_25 = RUNS_DIR / "best_25.pt"
    MODEL_50 = RUNS_DIR / "best_50.pt"
    MODEL_100 = RUNS_DIR / "best_100.pt"
    return RUN_100_DIR, RUN_500_DIR


@app.cell
def _(mo):
    mo.md("""## 1. Data Loading & Overview""")
    return


@app.cell
def _(RUN_100_DIR, RUN_500_DIR, pl):
    # Load results CSVs
    df_100 = pl.read_csv(RUN_100_DIR / "results.csv")
    df_500 = pl.read_csv(RUN_500_DIR / "results.csv")

    # Display first few rows
    df_100.head()
    return df_100, df_500


@app.cell
def _(mo):
    mo.md("""### Training Configuration Comparison""")
    return


@app.cell
def _(RUN_100_DIR, RUN_500_DIR, yaml):
    # Load hyperparameters from args.yaml
    with open(RUN_100_DIR / "args.yaml", "r") as f:
        args_100 = yaml.safe_load(f)

    with open(RUN_500_DIR / "args.yaml", "r") as f:
        args_500 = yaml.safe_load(f)

    # Key hyperparameters to compare
    key_params = [
        "epochs", "batch", "imgsz", "lr0", "lrf", "momentum",
        "weight_decay", "dropout", "optimizer", "mosaic", "mixup"
    ]

    print("=" * 80)
    print("HYPERPARAMETER COMPARISON")
    print("=" * 80)
    print(f"{'Parameter':<20} {'100-epoch':<20} {'500-epoch':<20} {'Match':<10}")
    print("-" * 80)

    for param in key_params:
        val_100 = args_100.get(param, "N/A")
        val_500 = args_500.get(param, "N/A")
        match = "✓" if val_100 == val_500 else "DIFF"
        print(f"{param:<20} {str(val_100):<20} {str(val_500):<20} {match:<10}")
    return


@app.cell
def _(mo):
    mo.md("""### Final Performance Metrics Comparison""")
    return


@app.cell
def _(df_100, df_500, pl):
    # Get final epoch metrics for comparison
    final_100 = df_100.tail(1)
    final_500 = df_500.tail(1)

    # Create comparison DataFrame
    metrics_comparison = pl.DataFrame({
        "Metric": [
            "Training Time (min)",
            "mAP50",
            "mAP50-95",
            "Precision",
            "Recall",
            "Train Box Loss",
            "Val Box Loss",
            "Train Cls Loss",
            "Val Cls Loss",
        ],
        "100 Epochs": [
            f"{final_100['time'][0] / 60:.1f}",
            f"{final_100['metrics/mAP50(B)'][0]:.4f}",
            f"{final_100['metrics/mAP50-95(B)'][0]:.4f}",
            f"{final_100['metrics/precision(B)'][0]:.4f}",
            f"{final_100['metrics/recall(B)'][0]:.4f}",
            f"{final_100['train/box_loss'][0]:.4f}",
            f"{final_100['val/box_loss'][0]:.4f}",
            f"{final_100['train/cls_loss'][0]:.4f}",
            f"{final_100['val/cls_loss'][0]:.4f}",
        ],
        "500 Epochs": [
            f"{final_500['time'][0] / 60:.1f}",
            f"{final_500['metrics/mAP50(B)'][0]:.4f}",
            f"{final_500['metrics/mAP50-95(B)'][0]:.4f}",
            f"{final_500['metrics/precision(B)'][0]:.4f}",
            f"{final_500['metrics/recall(B)'][0]:.4f}",
            f"{final_500['train/box_loss'][0]:.4f}",
            f"{final_500['val/box_loss'][0]:.4f}",
            f"{final_500['train/cls_loss'][0]:.4f}",
            f"{final_500['val/cls_loss'][0]:.4f}",
        ],
    })

    metrics_comparison
    return final_100, final_500


@app.cell
def _(final_100, final_500):
    # Calculate differences
    mAP50_diff = (final_500['metrics/mAP50(B)'][0] - final_100['metrics/mAP50(B)'][0]) * 100
    mAP5095_diff = (final_500['metrics/mAP50-95(B)'][0] - final_100['metrics/mAP50-95(B)'][0]) * 100
    precision_diff = (final_500['metrics/precision(B)'][0] - final_100['metrics/precision(B)'][0]) * 100
    recall_diff = (final_500['metrics/recall(B)'][0] - final_100['metrics/recall(B)'][0]) * 100

    train_loss_diff = final_500['train/box_loss'][0] - final_100['train/box_loss'][0]
    val_loss_diff = final_500['val/box_loss'][0] - final_100['val/box_loss'][0]

    print("=" * 80)
    print("PERFORMANCE DELTA (500 epochs - 100 epochs)")
    print("=" * 80)
    print(f"mAP50 change:          {mAP50_diff:+.2f}%")
    print(f"mAP50-95 change:       {mAP5095_diff:+.2f}%")
    print(f"Precision change:      {precision_diff:+.2f}%")
    print(f"Recall change:         {recall_diff:+.2f}%")
    print(f"\nTrain box loss change: {train_loss_diff:+.4f}")
    print(f"Val box loss change:   {val_loss_diff:+.4f}")

    if val_loss_diff > 0:
        print("\n⚠ WARNING: Validation loss INCREASED - potential overfitting detected!")
    else:
        print("\n✓ Validation loss decreased or stable")
    return (mAP50_diff,)


@app.cell
def _(mo):
    mo.md(
        """
    ## 2. Overfitting Analysis: Train vs Val Loss

    **Critical for Section 8**: Detect if extended training causes overfitting.
    """
    )
    return


@app.cell
def _(df_100, df_500, plt):
    _fig, _axes = plt.subplots(2, 3, figsize=(20, 12))

    # Box Loss Comparison
    _axes[0, 0].plot(df_100['epoch'], df_100['train/box_loss'], 'b-', label='Train (100)', linewidth=2)
    _axes[0, 0].plot(df_100['epoch'], df_100['val/box_loss'], 'b--', label='Val (100)', linewidth=2)
    _axes[0, 0].plot(df_500['epoch'], df_500['train/box_loss'], 'r-', label='Train (500)', linewidth=2, alpha=0.7)
    _axes[0, 0].plot(df_500['epoch'], df_500['val/box_loss'], 'r--', label='Val (500)', linewidth=2, alpha=0.7)
    _axes[0, 0].set_xlabel('Epoch', fontsize=12)
    _axes[0, 0].set_ylabel('Box Loss', fontsize=12)
    _axes[0, 0].set_title('Box Loss: Train vs Validation', fontsize=14, fontweight='bold')
    _axes[0, 0].legend(fontsize=10)
    _axes[0, 0].grid(True, alpha=0.3)
    _axes[0, 0].axvline(x=100, color='gray', linestyle=':', alpha=0.5, label='100 epochs')

    # Classification Loss Comparison
    _axes[0, 1].plot(df_100['epoch'], df_100['train/cls_loss'], 'b-', label='Train (100)', linewidth=2)
    _axes[0, 1].plot(df_100['epoch'], df_100['val/cls_loss'], 'b--', label='Val (100)', linewidth=2)
    _axes[0, 1].plot(df_500['epoch'], df_500['train/cls_loss'], 'r-', label='Train (500)', linewidth=2, alpha=0.7)
    _axes[0, 1].plot(df_500['epoch'], df_500['val/cls_loss'], 'r--', label='Val (500)', linewidth=2, alpha=0.7)
    _axes[0, 1].set_xlabel('Epoch', fontsize=12)
    _axes[0, 1].set_ylabel('Classification Loss', fontsize=12)
    _axes[0, 1].set_title('Classification Loss: Train vs Validation', fontsize=14, fontweight='bold')
    _axes[0, 1].legend(fontsize=10)
    _axes[0, 1].grid(True, alpha=0.3)
    _axes[0, 1].axvline(x=100, color='gray', linestyle=':', alpha=0.5)

    # DFL Loss Comparison
    _axes[0, 2].plot(df_100['epoch'], df_100['train/dfl_loss'], 'b-', label='Train (100)', linewidth=2)
    _axes[0, 2].plot(df_100['epoch'], df_100['val/dfl_loss'], 'b--', label='Val (100)', linewidth=2)
    _axes[0, 2].plot(df_500['epoch'], df_500['train/dfl_loss'], 'r-', label='Train (500)', linewidth=2, alpha=0.7)
    _axes[0, 2].plot(df_500['epoch'], df_500['val/dfl_loss'], 'r--', label='Val (500)', linewidth=2, alpha=0.7)
    _axes[0, 2].set_xlabel('Epoch', fontsize=12)
    _axes[0, 2].set_ylabel('DFL Loss', fontsize=12)
    _axes[0, 2].set_title('DFL Loss: Train vs Validation', fontsize=14, fontweight='bold')
    _axes[0, 2].legend(fontsize=10)
    _axes[0, 2].grid(True, alpha=0.3)
    _axes[0, 2].axvline(x=100, color='gray', linestyle=':', alpha=0.5)

    # Train-Val Gap for Box Loss
    gap_100 = df_100['val/box_loss'] - df_100['train/box_loss']
    gap_500 = df_500['val/box_loss'] - df_500['train/box_loss']

    _axes[1, 0].plot(df_100['epoch'], gap_100, 'b-', label='100 epochs', linewidth=2)
    _axes[1, 0].plot(df_500['epoch'], gap_500, 'r-', label='500 epochs', linewidth=2, alpha=0.7)
    _axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=1)
    _axes[1, 0].set_xlabel('Epoch', fontsize=12)
    _axes[1, 0].set_ylabel('Val Loss - Train Loss', fontsize=12)
    _axes[1, 0].set_title('Overfitting Indicator: Val-Train Gap (Box)', fontsize=14, fontweight='bold')
    _axes[1, 0].legend(fontsize=10)
    _axes[1, 0].grid(True, alpha=0.3)
    _axes[1, 0].axvline(x=100, color='gray', linestyle=':', alpha=0.5)
    _axes[1, 0].fill_between(df_100['epoch'], 0, gap_100, where=(gap_100 > 0), alpha=0.2, color='red', label='Overfitting region')

    # Train-Val Gap for Classification Loss
    gap_cls_100 = df_100['val/cls_loss'] - df_100['train/cls_loss']
    gap_cls_500 = df_500['val/cls_loss'] - df_500['train/cls_loss']

    _axes[1, 1].plot(df_100['epoch'], gap_cls_100, 'b-', label='100 epochs', linewidth=2)
    _axes[1, 1].plot(df_500['epoch'], gap_cls_500, 'r-', label='500 epochs', linewidth=2, alpha=0.7)
    _axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=1)
    _axes[1, 1].set_xlabel('Epoch', fontsize=12)
    _axes[1, 1].set_ylabel('Val Loss - Train Loss', fontsize=12)
    _axes[1, 1].set_title('Overfitting Indicator: Val-Train Gap (Cls)', fontsize=14, fontweight='bold')
    _axes[1, 1].legend(fontsize=10)
    _axes[1, 1].grid(True, alpha=0.3)
    _axes[1, 1].axvline(x=100, color='gray', linestyle=':', alpha=0.5)

    # Combined Total Loss
    total_train_100 = df_100['train/box_loss'] + df_100['train/cls_loss'] + df_100['train/dfl_loss']
    total_val_100 = df_100['val/box_loss'] + df_100['val/cls_loss'] + df_100['val/dfl_loss']
    total_train_500 = df_500['train/box_loss'] + df_500['train/cls_loss'] + df_500['train/dfl_loss']
    total_val_500 = df_500['val/box_loss'] + df_500['val/cls_loss'] + df_500['val/dfl_loss']

    _axes[1, 2].plot(df_100['epoch'], total_train_100, 'b-', label='Train (100)', linewidth=2)
    _axes[1, 2].plot(df_100['epoch'], total_val_100, 'b--', label='Val (100)', linewidth=2)
    _axes[1, 2].plot(df_500['epoch'], total_train_500, 'r-', label='Train (500)', linewidth=2, alpha=0.7)
    _axes[1, 2].plot(df_500['epoch'], total_val_500, 'r--', label='Val (500)', linewidth=2, alpha=0.7)
    _axes[1, 2].set_xlabel('Epoch', fontsize=12)
    _axes[1, 2].set_ylabel('Total Loss', fontsize=12)
    _axes[1, 2].set_title('Combined Total Loss', fontsize=14, fontweight='bold')
    _axes[1, 2].legend(fontsize=10)
    _axes[1, 2].grid(True, alpha=0.3)
    _axes[1, 2].axvline(x=100, color='gray', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 3. Performance Metrics Evolution

    Track how mAP, precision, and recall change over training.
    """
    )
    return


@app.cell
def _(df_100, df_500, plt):
    _fig, _axes = plt.subplots(2, 2, figsize=(18, 12))

    # mAP50 Comparison
    _axes[0, 0].plot(df_100['epoch'], df_100['metrics/mAP50(B)'], 'b-', label='100 epochs', linewidth=2.5, marker='o', markersize=3)
    _axes[0, 0].plot(df_500['epoch'], df_500['metrics/mAP50(B)'], 'r-', label='500 epochs', linewidth=2.5, marker='o', markersize=2, alpha=0.7)

    # Mark best mAP50 for each
    best_idx_100 = df_100['metrics/mAP50(B)'].arg_max()
    best_idx_500 = df_500['metrics/mAP50(B)'].arg_max()
    _axes[0, 0].scatter(df_100['epoch'][best_idx_100], df_100['metrics/mAP50(B)'][best_idx_100],
                       color='blue', s=200, marker='*', zorder=5, label=f'Best 100: epoch {df_100["epoch"][best_idx_100]}')
    _axes[0, 0].scatter(df_500['epoch'][best_idx_500], df_500['metrics/mAP50(B)'][best_idx_500],
                       color='red', s=200, marker='*', zorder=5, label=f'Best 500: epoch {df_500["epoch"][best_idx_500]}')

    _axes[0, 0].set_xlabel('Epoch', fontsize=12)
    _axes[0, 0].set_ylabel('mAP50', fontsize=12)
    _axes[0, 0].set_title('mAP@0.5 Progression', fontsize=14, fontweight='bold')
    _axes[0, 0].legend(fontsize=10)
    _axes[0, 0].grid(True, alpha=0.3)
    _axes[0, 0].axvline(x=100, color='gray', linestyle=':', alpha=0.5)

    # mAP50-95 Comparison
    _axes[0, 1].plot(df_100['epoch'], df_100['metrics/mAP50-95(B)'], 'b-', label='100 epochs', linewidth=2.5, marker='o', markersize=3)
    _axes[0, 1].plot(df_500['epoch'], df_500['metrics/mAP50-95(B)'], 'r-', label='500 epochs', linewidth=2.5, marker='o', markersize=2, alpha=0.7)
    _axes[0, 1].set_xlabel('Epoch', fontsize=12)
    _axes[0, 1].set_ylabel('mAP50-95', fontsize=12)
    _axes[0, 1].set_title('mAP@[0.5:0.95] Progression', fontsize=14, fontweight='bold')
    _axes[0, 1].legend(fontsize=10)
    _axes[0, 1].grid(True, alpha=0.3)
    _axes[0, 1].axvline(x=100, color='gray', linestyle=':', alpha=0.5)

    # Precision Comparison
    _axes[1, 0].plot(df_100['epoch'], df_100['metrics/precision(B)'], 'b-', label='100 epochs', linewidth=2.5, marker='o', markersize=3)
    _axes[1, 0].plot(df_500['epoch'], df_500['metrics/precision(B)'], 'r-', label='500 epochs', linewidth=2.5, marker='o', markersize=2, alpha=0.7)
    _axes[1, 0].set_xlabel('Epoch', fontsize=12)
    _axes[1, 0].set_ylabel('Precision', fontsize=12)
    _axes[1, 0].set_title('Precision Progression', fontsize=14, fontweight='bold')
    _axes[1, 0].legend(fontsize=10)
    _axes[1, 0].grid(True, alpha=0.3)
    _axes[1, 0].axvline(x=100, color='gray', linestyle=':', alpha=0.5)

    # Recall Comparison
    _axes[1, 1].plot(df_100['epoch'], df_100['metrics/recall(B)'], 'b-', label='100 epochs', linewidth=2.5, marker='o', markersize=3)
    _axes[1, 1].plot(df_500['epoch'], df_500['metrics/recall(B)'], 'r-', label='500 epochs', linewidth=2.5, marker='o', markersize=2, alpha=0.7)
    _axes[1, 1].set_xlabel('Epoch', fontsize=12)
    _axes[1, 1].set_ylabel('Recall', fontsize=12)
    _axes[1, 1].set_title('Recall Progression', fontsize=14, fontweight='bold')
    _axes[1, 1].legend(fontsize=10)
    _axes[1, 1].grid(True, alpha=0.3)
    _axes[1, 1].axvline(x=100, color='gray', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.gca()
    return best_idx_100, best_idx_500


@app.cell
def _(best_idx_100, best_idx_500, df_100, df_500):
    print("=" * 80)
    print("BEST PERFORMANCE EPOCHS")
    print("=" * 80)
    print(f"\n100-Epoch Training:")
    print(f"  Best mAP50 at epoch: {df_100['epoch'][best_idx_100]}")
    print(f"  Best mAP50 value: {df_100['metrics/mAP50(B)'][best_idx_100]:.4f}")

    print(f"\n500-Epoch Training:")
    print(f"  Best mAP50 at epoch: {df_500['epoch'][best_idx_500]}")
    print(f"  Best mAP50 value: {df_500['metrics/mAP50(B)'][best_idx_500]:.4f}")

    print(f"\nImprovement: {(df_500['metrics/mAP50(B)'][best_idx_500] - df_100['metrics/mAP50(B)'][best_idx_100]) * 100:.2f}%")

    if best_idx_500 < 100:
        print("\n⚠ INSIGHT: Best performance achieved BEFORE 100 epochs in 500-epoch run")
        print("   → Early stopping would have been beneficial")
    elif best_idx_500 > 100:
        print(f"\n✓ INSIGHT: Extended training improved performance")
        print(f"   → Best epoch was {df_500['epoch'][best_idx_500]}, gaining {(df_500['metrics/mAP50(B)'][best_idx_500] - df_100['metrics/mAP50(B)'][best_idx_100]) * 100:.2f}% mAP50")
    return


@app.cell
def _(mo):
    mo.md("""## 4. Precision-Recall Tradeoff Analysis""")
    return


@app.cell
def _(df_100, df_500, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(18, 6))

    # Precision vs Recall trajectory
    _axes[0].plot(df_100['metrics/recall(B)'], df_100['metrics/precision(B)'],
                 'b-', linewidth=2.5, marker='o', markersize=4, label='100 epochs')
    _axes[0].plot(df_500['metrics/recall(B)'], df_500['metrics/precision(B)'],
                 'r-', linewidth=2.5, marker='o', markersize=3, alpha=0.7, label='500 epochs')

    # Mark start and end
    _axes[0].scatter(df_100['metrics/recall(B)'][0], df_100['metrics/precision(B)'][0],
                    color='blue', s=150, marker='s', zorder=5, label='Start (100)')
    _axes[0].scatter(df_100['metrics/recall(B)'][-1], df_100['metrics/precision(B)'][-1],
                    color='blue', s=150, marker='*', zorder=5, label='End (100)')
    _axes[0].scatter(df_500['metrics/recall(B)'][0], df_500['metrics/precision(B)'][0],
                    color='red', s=150, marker='s', zorder=5, label='Start (500)')
    _axes[0].scatter(df_500['metrics/recall(B)'][-1], df_500['metrics/precision(B)'][-1],
                    color='red', s=150, marker='*', zorder=5, label='End (500)')

    _axes[0].set_xlabel('Recall', fontsize=12)
    _axes[0].set_ylabel('Precision', fontsize=12)
    _axes[0].set_title('Precision-Recall Trajectory', fontsize=14, fontweight='bold')
    _axes[0].legend(fontsize=9)
    _axes[0].grid(True, alpha=0.3)
    _axes[0].set_xlim([0, 1])
    _axes[0].set_ylim([0, 1])

    # F1 Score over epochs
    f1_100 = 2 * (df_100['metrics/precision(B)'] * df_100['metrics/recall(B)']) / (df_100['metrics/precision(B)'] + df_100['metrics/recall(B)'])
    f1_500 = 2 * (df_500['metrics/precision(B)'] * df_500['metrics/recall(B)']) / (df_500['metrics/precision(B)'] + df_500['metrics/recall(B)'])

    _axes[1].plot(df_100['epoch'], f1_100, 'b-', linewidth=2.5, marker='o', markersize=3, label='100 epochs')
    _axes[1].plot(df_500['epoch'], f1_500, 'r-', linewidth=2.5, marker='o', markersize=2, alpha=0.7, label='500 epochs')

    # Mark best F1
    best_f1_100_idx = f1_100.arg_max()
    best_f1_500_idx = f1_500.arg_max()
    _axes[1].scatter(df_100['epoch'][best_f1_100_idx], f1_100[best_f1_100_idx],
                    color='blue', s=200, marker='*', zorder=5)
    _axes[1].scatter(df_500['epoch'][best_f1_500_idx], f1_500[best_f1_500_idx],
                    color='red', s=200, marker='*', zorder=5)

    _axes[1].set_xlabel('Epoch', fontsize=12)
    _axes[1].set_ylabel('F1 Score', fontsize=12)
    _axes[1].set_title('F1 Score Progression', fontsize=14, fontweight='bold')
    _axes[1].legend(fontsize=10)
    _axes[1].grid(True, alpha=0.3)
    _axes[1].axvline(x=100, color='gray', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.gca()
    return best_f1_100_idx, best_f1_500_idx, f1_100, f1_500


@app.cell
def _(best_f1_100_idx, best_f1_500_idx, df_100, df_500, f1_100, f1_500):
    print("=" * 80)
    print("F1 SCORE ANALYSIS")
    print("=" * 80)
    print(f"\n100-Epoch Training:")
    print(f"  Best F1: {f1_100[best_f1_100_idx]:.4f} at epoch {df_100['epoch'][best_f1_100_idx]}")
    print(f"  Final F1: {f1_100[-1]:.4f}")

    print(f"\n500-Epoch Training:")
    print(f"  Best F1: {f1_500[best_f1_500_idx]:.4f} at epoch {df_500['epoch'][best_f1_500_idx]}")
    print(f"  Final F1: {f1_500[-1]:.4f}")

    print(f"\nImprovement in best F1: {(f1_500[best_f1_500_idx] - f1_100[best_f1_100_idx]) * 100:.2f}%")
    return


@app.cell
def _(mo):
    mo.md("""## 5. Learning Rate Schedule Analysis""")
    return


@app.cell
def _(df_100, df_500, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(18, 6))

    # Learning rate schedule
    _axes[0].plot(df_100['epoch'], df_100['lr/pg0'], 'b-', linewidth=2, label='100 epochs')
    _axes[0].plot(df_500['epoch'], df_500['lr/pg0'], 'r-', linewidth=2, alpha=0.7, label='500 epochs')
    _axes[0].set_xlabel('Epoch', fontsize=12)
    _axes[0].set_ylabel('Learning Rate', fontsize=12)
    _axes[0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    _axes[0].legend(fontsize=10)
    _axes[0].grid(True, alpha=0.3)
    _axes[0].axvline(x=100, color='gray', linestyle=':', alpha=0.5)

    # mAP50 vs Learning Rate
    _axes[1].scatter(df_100['lr/pg0'], df_100['metrics/mAP50(B)'],
                    c=df_100['epoch'], cmap='Blues', s=50, alpha=0.6, label='100 epochs')
    _axes[1].scatter(df_500['lr/pg0'], df_500['metrics/mAP50(B)'],
                    c=df_500['epoch'], cmap='Reds', s=30, alpha=0.6, label='500 epochs')
    _axes[1].set_xlabel('Learning Rate', fontsize=12)
    _axes[1].set_ylabel('mAP50', fontsize=12)
    _axes[1].set_title('mAP50 vs Learning Rate', fontsize=14, fontweight='bold')
    _axes[1].legend(fontsize=10)
    _axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 6. Convergence Analysis

    Identify when the model converges and if extended training helps.
    """
    )
    return


@app.cell
def _(df_100, df_500, np, plt):
    # Calculate moving average of mAP50 to smooth curves
    window = 5

    def moving_avg(data, window):
        return np.convolve(data.to_numpy(), np.ones(window)/window, mode='valid')

    ma_100 = moving_avg(df_100['metrics/mAP50(B)'], window)
    ma_500 = moving_avg(df_500['metrics/mAP50(B)'], window)

    # Calculate improvement rate (derivative)
    improvement_100 = np.diff(ma_100)
    improvement_500 = np.diff(ma_500)

    _fig, _axes = plt.subplots(1, 2, figsize=(18, 6))

    # Smoothed mAP50
    _axes[0].plot(range(window, len(df_100) + 1), ma_100, 'b-', linewidth=2.5, label='100 epochs (smoothed)')
    _axes[0].plot(range(window, len(df_500) + 1), ma_500, 'r-', linewidth=2.5, alpha=0.7, label='500 epochs (smoothed)')
    _axes[0].set_xlabel('Epoch', fontsize=12)
    _axes[0].set_ylabel('mAP50 (Moving Avg)', fontsize=12)
    _axes[0].set_title(f'Smoothed mAP50 (window={window})', fontsize=14, fontweight='bold')
    _axes[0].legend(fontsize=10)
    _axes[0].grid(True, alpha=0.3)
    _axes[0].axvline(x=100, color='gray', linestyle=':', alpha=0.5)

    # Improvement rate
    _axes[1].plot(range(window + 1, len(df_100) + 1), improvement_100, 'b-', linewidth=2, label='100 epochs')
    _axes[1].plot(range(window + 1, len(df_500) + 1), improvement_500, 'r-', linewidth=2, alpha=0.7, label='500 epochs')
    _axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
    _axes[1].set_xlabel('Epoch', fontsize=12)
    _axes[1].set_ylabel('Improvement Rate (dmAP50/depoch)', fontsize=12)
    _axes[1].set_title('Convergence Rate Analysis', fontsize=14, fontweight='bold')
    _axes[1].legend(fontsize=10)
    _axes[1].grid(True, alpha=0.3)
    _axes[1].axvline(x=100, color='gray', linestyle=':', alpha=0.5)

    # Find convergence point (when improvement < threshold)
    threshold = 0.001
    converged_100 = np.where(np.abs(improvement_100) < threshold)[0]
    converged_500 = np.where(np.abs(improvement_500) < threshold)[0]

    if len(converged_100) > 0:
        first_converge_100 = converged_100[0] + window + 1
        _axes[1].axvline(x=first_converge_100, color='blue', linestyle='--', alpha=0.5,
                        label=f'Converged at epoch {first_converge_100}')

    if len(converged_500) > 0:
        first_converge_500 = converged_500[0] + window + 1
        _axes[1].axvline(x=first_converge_500, color='red', linestyle='--', alpha=0.5,
                        label=f'Converged at epoch {first_converge_500}')

    _axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.gca()
    return converged_100, converged_500, first_converge_100, first_converge_500


@app.cell
def _(converged_100, converged_500, first_converge_100, first_converge_500):
    print("=" * 80)
    print("CONVERGENCE ANALYSIS")
    print("=" * 80)

    if len(converged_100) > 0:
        print(f"\n100-Epoch Training:")
        print(f"  First convergence at epoch: {first_converge_100}")
        print(f"  Percentage of training after convergence: {((100 - first_converge_100) / 100) * 100:.1f}%")
    else:
        print(f"\n100-Epoch Training:")
        print(f"  Did not fully converge (still improving)")

    if len(converged_500) > 0:
        print(f"\n500-Epoch Training:")
        print(f"  First convergence at epoch: {first_converge_500}")
        print(f"  Percentage of training after convergence: {((500 - first_converge_500) / 500) * 100:.1f}%")
    else:
        print(f"\n500-Epoch Training:")
        print(f"  Did not fully converge (still improving)")
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 7. Visual Results Comparison

    Compare confusion matrices and training plots side-by-side.
    """
    )
    return


@app.cell
def _(RUN_100_DIR, RUN_500_DIR, mpimg, plt):
    _fig, _axes = plt.subplots(2, 2, figsize=(20, 20))

    # Confusion matrices
    cm_100 = mpimg.imread(str(RUN_100_DIR / "confusion_matrix_normalized.png"))
    cm_500 = mpimg.imread(str(RUN_500_DIR / "confusion_matrix_normalized.png"))

    _axes[0, 0].imshow(cm_100)
    _axes[0, 0].set_title('Confusion Matrix: 100 Epochs', fontsize=14, fontweight='bold')
    _axes[0, 0].axis('off')

    _axes[0, 1].imshow(cm_500)
    _axes[0, 1].set_title('Confusion Matrix: 500 Epochs', fontsize=14, fontweight='bold')
    _axes[0, 1].axis('off')

    # Results plots
    results_100 = mpimg.imread(str(RUN_100_DIR / "results.png"))
    results_500 = mpimg.imread(str(RUN_500_DIR / "results.png"))

    _axes[1, 0].imshow(results_100)
    _axes[1, 0].set_title('Training Metrics: 100 Epochs', fontsize=14, fontweight='bold')
    _axes[1, 0].axis('off')

    _axes[1, 1].imshow(results_500)
    _axes[1, 1].set_title('Training Metrics: 500 Epochs', fontsize=14, fontweight='bold')
    _axes[1, 1].axis('off')

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md("""## 8. Precision-Recall Curves Comparison""")
    return


@app.cell
def _(RUN_100_DIR, RUN_500_DIR, mpimg, plt):
    _fig, _axes = plt.subplots(2, 3, figsize=(24, 16))

    # Load all curves
    pr_100 = mpimg.imread(str(RUN_100_DIR / "BoxPR_curve.png"))
    p_100 = mpimg.imread(str(RUN_100_DIR / "BoxP_curve.png"))
    r_100 = mpimg.imread(str(RUN_100_DIR / "BoxR_curve.png"))

    pr_500 = mpimg.imread(str(RUN_500_DIR / "BoxPR_curve.png"))
    p_500 = mpimg.imread(str(RUN_500_DIR / "BoxP_curve.png"))
    r_500 = mpimg.imread(str(RUN_500_DIR / "BoxR_curve.png"))

    _axes[0, 0].imshow(pr_100)
    _axes[0, 0].set_title('P-R Curve: 100 Epochs', fontsize=13, fontweight='bold')
    _axes[0, 0].axis('off')

    _axes[0, 1].imshow(p_100)
    _axes[0, 1].set_title('Precision Curve: 100 Epochs', fontsize=13, fontweight='bold')
    _axes[0, 1].axis('off')

    _axes[0, 2].imshow(r_100)
    _axes[0, 2].set_title('Recall Curve: 100 Epochs', fontsize=13, fontweight='bold')
    _axes[0, 2].axis('off')

    _axes[1, 0].imshow(pr_500)
    _axes[1, 0].set_title('P-R Curve: 500 Epochs', fontsize=13, fontweight='bold')
    _axes[1, 0].axis('off')

    _axes[1, 1].imshow(p_500)
    _axes[1, 1].set_title('Precision Curve: 500 Epochs', fontsize=13, fontweight='bold')
    _axes[1, 1].axis('off')

    _axes[1, 2].imshow(r_500)
    _axes[1, 2].set_title('Recall Curve: 500 Epochs', fontsize=13, fontweight='bold')
    _axes[1, 2].axis('off')

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md("""## 9. F1 Curves Comparison""")
    return


@app.cell
def _(RUN_100_DIR, RUN_500_DIR, mpimg, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(20, 8))

    _f1_img_100 = mpimg.imread(str(RUN_100_DIR / "BoxF1_curve.png"))
    _f1_img_500 = mpimg.imread(str(RUN_500_DIR / "BoxF1_curve.png"))

    _axes[0].imshow(_f1_img_100)
    _axes[0].set_title('F1-Confidence Curve: 100 Epochs', fontsize=14, fontweight='bold')
    _axes[0].axis('off')

    _axes[1].imshow(_f1_img_500)
    _axes[1].set_title('F1-Confidence Curve: 500 Epochs', fontsize=14, fontweight='bold')
    _axes[1].axis('off')

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 10. Statistical Analysis

    Test if performance differences are statistically significant.
    """
    )
    return


@app.cell
def _(df_100, df_500, np, stats):
    # Take last 20 epochs to compare stable performance
    last_n = 20

    mAP50_100_stable = df_100['metrics/mAP50(B)'][-last_n:].to_numpy()
    mAP50_500_stable = df_500['metrics/mAP50(B)'][-last_n:].to_numpy()

    # Perform t-test
    t_stat, p_value = stats.ttest_ind(mAP50_100_stable, mAP50_500_stable)

    print("=" * 80)
    print("STATISTICAL SIGNIFICANCE TEST (Independent t-test)")
    print("=" * 80)
    print(f"\nComparing final {last_n} epochs of each training run:")
    print(f"\n100-Epoch Training (last {last_n} epochs):")
    print(f"  Mean mAP50: {np.mean(mAP50_100_stable):.4f}")
    print(f"  Std Dev: {np.std(mAP50_100_stable):.4f}")
    print(f"  Min: {np.min(mAP50_100_stable):.4f}, Max: {np.max(mAP50_100_stable):.4f}")

    print(f"\n500-Epoch Training (last {last_n} epochs):")
    print(f"  Mean mAP50: {np.mean(mAP50_500_stable):.4f}")
    print(f"  Std Dev: {np.std(mAP50_500_stable):.4f}")
    print(f"  Min: {np.min(mAP50_500_stable):.4f}, Max: {np.max(mAP50_500_stable):.4f}")

    print(f"\nT-test Results:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")

    alpha = 0.05
    if p_value < alpha:
        print(f"\n✓ SIGNIFICANT: p < {alpha}")
        print("  The performance difference between 100 and 500 epochs IS statistically significant.")
    else:
        print(f"\n✗ NOT SIGNIFICANT: p >= {alpha}")
        print("  The performance difference between 100 and 500 epochs is NOT statistically significant.")
        print("  Extended training may not provide meaningful improvement.")
    return (p_value,)


@app.cell
def _(mo):
    mo.md(
        """
    ## 11. Training Efficiency Analysis

    Compare time investment vs performance gain.
    """
    )
    return


@app.cell
def _(df_500, final_100, final_500, pl):
    time_100 = final_100['time'][0] / 3600
    time_500 = final_500['time'][0] / 3600

    mAP_100 = final_100['metrics/mAP50(B)'][0]
    mAP_500 = final_500['metrics/mAP50(B)'][0]

    time_per_epoch_100 = time_100 / 100
    time_per_epoch_500 = time_500 / 500

    mAP_gain = (mAP_500 - mAP_100) * 100
    time_cost = time_500 - time_100
    efficiency = mAP_gain / time_cost if time_cost > 0 else 0

    print("=" * 80)
    print("TRAINING EFFICIENCY ANALYSIS")
    print("=" * 80)
    print(f"\n100-Epoch Training:")
    print(f"  Total time: {time_100:.2f} hours")
    print(f"  Time per epoch: {time_per_epoch_100 * 60:.2f} minutes")
    print(f"  Final mAP50: {mAP_100:.4f}")

    print(f"\n500-Epoch Training:")
    print(f"  Total time: {time_500:.2f} hours")
    print(f"  Time per epoch: {time_per_epoch_500 * 60:.2f} minutes")
    print(f"  Final mAP50: {mAP_500:.4f}")

    print(f"\nCost-Benefit Analysis:")
    print(f"  Additional training time: {time_cost:.2f} hours ({time_cost * 60:.1f} minutes)")
    print(f"  mAP50 gain: {mAP_gain:.2f}%")
    print(f"  Efficiency: {efficiency:.4f}% mAP gain per hour")

    # Calculate epochs needed to reach 100-epoch performance in 500-epoch run
    exceeds = df_500.filter(pl.col('metrics/mAP50(B)') >= mAP_100)
    if len(exceeds) > 0:
        epoch_to_match = exceeds['epoch'][0]
        time_to_match = exceeds['time'][0] / 3600
        print(f"\n✓ INSIGHT: 500-epoch training reached 100-epoch performance at:")
        print(f"  Epoch {epoch_to_match} ({time_to_match:.2f} hours)")
        print(f"  This suggests ~{epoch_to_match} epochs may be sufficient")
    else:
        epoch_to_match = None
        time_to_match = None
        print(f"\n⚠ 500-epoch training did not reach 100-epoch final performance in early epochs")
    return efficiency, epoch_to_match


@app.cell
def _(mo):
    mo.md(
        """
    ## 12. Key Observations & Recommendations for Report

    **Section 8 (Impact of Hyperparameters) - Critical Insights**
    """
    )
    return


@app.cell
def _(
    converged_100,
    converged_500,
    df_100,
    df_500,
    efficiency,
    epoch_to_match,
    final_100,
    final_500,
    first_converge_100,
    first_converge_500,
    mAP50_diff,
    p_value,
):
    print("=" * 80)
    print("FINAL OBSERVATIONS FOR CS4287 ASSIGNMENT SECTION 8")
    print("=" * 80)

    print("\n1. OVERFITTING DETECTION:")

    # Check for overfitting signs
    train_val_gap_100 = final_100['val/box_loss'][0] - final_100['train/box_loss'][0]
    train_val_gap_500 = final_500['val/box_loss'][0] - final_500['train/box_loss'][0]

    print(f"   100 epochs - Train/Val gap: {train_val_gap_100:.4f}")
    print(f"   500 epochs - Train/Val gap: {train_val_gap_500:.4f}")

    if train_val_gap_500 > train_val_gap_100:
        print("   ⚠ OVERFITTING DETECTED: Val loss gap increased with extended training")
        print("   → Recommendation: Use early stopping around epoch", end=" ")
        if len(converged_100) > 0:
            print(f"{first_converge_100}-{first_converge_100 + 20}")
        else:
            print("80-100")
    else:
        print("   ✓ NO OVERFITTING: Model generalization maintained")

    print("\n2. CONVERGENCE ANALYSIS:")
    if len(converged_500) > 0:
        print(f"   Model converged at epoch {first_converge_500}")
        print(f"   Training beyond this point shows diminishing returns")
    else:
        print("   Model continued improving throughout 500 epochs")
        print("   Additional training may still be beneficial")

    print("\n3. PERFORMANCE vs TRAINING TIME TRADEOFF:")
    print(f"   mAP50 improvement: {mAP50_diff:.2f}%")
    print(f"   Efficiency: {efficiency:.4f}% mAP per hour")
    if mAP50_diff < 1.0 and efficiency < 0.1:
        print("   → RECOMMENDATION: Diminishing returns - 100 epochs sufficient")
    else:
        print("   → RECOMMENDATION: Extended training provides meaningful improvement")

    print("\n4. STATISTICAL SIGNIFICANCE:")
    if p_value < 0.05:
        print(f"   Performance difference IS statistically significant (p={p_value:.4f})")
        print("   → Extended training provides reliable improvement")
    else:
        print(f"   Performance difference NOT statistically significant (p={p_value:.4f})")
        print("   → Improvement may be due to random variation")

    print("\n5. OPTIMAL TRAINING CONFIGURATION:")
    best_epoch_500 = df_500['metrics/mAP50(B)'].arg_max()
    best_epoch_100 = df_100['metrics/mAP50(B)'].arg_max()
    print(f"   Best performance in 100-epoch run: Epoch {df_100['epoch'][best_epoch_100]} (mAP50: {df_100['metrics/mAP50(B)'][best_epoch_100]:.4f})")
    print(f"   Best performance in 500-epoch run: Epoch {df_500['epoch'][best_epoch_500]} (mAP50: {df_500['metrics/mAP50(B)'][best_epoch_500]:.4f})")

    if epoch_to_match:
        print(f"\n6. EARLY STOPPING RECOMMENDATION:")
        print(f"   Could stop at epoch ~{epoch_to_match} to match 100-epoch performance faster")
        print(f"   Then continue if validation performance still improving")

    print("\n" + "=" * 80)
    print("HYPERPARAMETER EXPERIMENTS TO ADD (for full Section 8 marks):")
    print("=" * 80)
    print("\nCurrent: Only EPOCHS varied (10, 25, 50, 100, 500)")
    print("\nRecommended additional experiments:")
    print("  1. Learning Rate: Try lr0 = [0.001, 0.005, 0.02]")
    print("  2. Batch Size: Try batch = [8, 32] to compare with current 16")
    print("  3. Image Size: Try imgsz = [416, 800] to compare with current 640")
    print("  4. Optimizer: Try optimizer = 'SGD' vs current 'auto' (AdamW)")
    print("  5. Dropout: Try dropout = [0.1, 0.2] vs current 0.0")
    print("  6. Data Augmentation: Try mosaic = 0.0 vs current 1.0")
    print("\nEach experiment should run for 100 epochs and be compared to baseline.")
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 13. Export Summary for Report

    Key metrics table for direct inclusion in LaTeX report.
    """
    )
    return


@app.cell
def _(
    df_100,
    df_500,
    efficiency,
    final_100,
    final_500,
    mAP50_diff,
    p_value,
    pl,
):
    # Create LaTeX-ready summary table
    best_idx_100_final = df_100['metrics/mAP50(B)'].arg_max()
    best_idx_500_final = df_500['metrics/mAP50(B)'].arg_max()

    report_summary = pl.DataFrame({
        "Configuration": ["100 Epochs", "500 Epochs", "Improvement"],
        "Training Time": [
            f"{final_100['time'][0] / 3600:.2f}h",
            f"{final_500['time'][0] / 3600:.2f}h",
            f"+{(final_500['time'][0] - final_100['time'][0]) / 3600:.2f}h"
        ],
        "Best Epoch": [
            f"{df_100['epoch'][best_idx_100_final]}",
            f"{df_500['epoch'][best_idx_500_final]}",
            "-"
        ],
        "Best mAP50": [
            f"{df_100['metrics/mAP50(B)'][best_idx_100_final]:.4f}",
            f"{df_500['metrics/mAP50(B)'][best_idx_500_final]:.4f}",
            f"{mAP50_diff:+.2f}%"
        ],
        "Final Precision": [
            f"{final_100['metrics/precision(B)'][0]:.4f}",
            f"{final_500['metrics/precision(B)'][0]:.4f}",
            f"{(final_500['metrics/precision(B)'][0] - final_100['metrics/precision(B)'][0]) * 100:+.2f}%"
        ],
        "Final Recall": [
            f"{final_100['metrics/recall(B)'][0]:.4f}",
            f"{final_500['metrics/recall(B)'][0]:.4f}",
            f"{(final_500['metrics/recall(B)'][0] - final_100['metrics/recall(B)'][0]) * 100:+.2f}%"
        ],
        "Efficiency": [
            "-",
            f"{efficiency:.4f}%/h",
            "-"
        ],
        "Stat. Sig. (p)": [
            "-",
            f"{p_value:.4f}",
            "Yes" if p_value < 0.05 else "No"
        ]
    })

    print("=" * 80)
    print("SUMMARY TABLE FOR REPORT (Section 6 & 8)")
    print("=" * 80)
    report_summary
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 14. Per-Class Performance Analysis

    Analyze performance for each class individually to identify problem areas.
    """
    )
    return


@app.cell
def _(RUN_100_DIR, RUN_500_DIR, pl):
    # Extract per-class metrics from confusion matrices
    def analyze_confusion_matrix(run_dir, run_name):
        if not run_dir.exists():
            return None

        # For now, use results from CSV (YOLO provides per-class metrics differently)
        _results = pl.read_csv(run_dir / "results.csv")
        _final_row = _results.tail(1)

        return {
            "run": run_name,
            "precision": _final_row["metrics/precision(B)"][0],
            "recall": _final_row["metrics/recall(B)"][0],
            "mAP50": _final_row["metrics/mAP50(B)"][0],
            "mAP50-95": _final_row["metrics/mAP50-95(B)"][0],
        }

    analysis_100 = analyze_confusion_matrix(RUN_100_DIR, "100 epochs")
    analysis_500 = analyze_confusion_matrix(RUN_500_DIR, "500 epochs")

    print("=" * 80)
    print("PER-CLASS PERFORMANCE SUMMARY")
    print("=" * 80)
    print("\nNote: YOLO provides overall metrics. For per-class analysis, see confusion matrices.")
    print(f"\n100-Epoch Run:")
    print(f"  Overall Precision: {analysis_100['precision']:.4f}")
    print(f"  Overall Recall: {analysis_100['recall']:.4f}")
    print(f"  Overall mAP50: {analysis_100['mAP50']:.4f}")

    print(f"\n500-Epoch Run:")
    print(f"  Overall Precision: {analysis_500['precision']:.4f}")
    print(f"  Overall Recall: {analysis_500['recall']:.4f}")
    print(f"  Overall mAP50: {analysis_500['mAP50']:.4f}")
    return


@app.cell
def _(mo):
    mo.md("""## 15. Confusion Matrix Delta Analysis""")
    return


@app.cell
def _(RUN_100_DIR, RUN_500_DIR, mpimg, plt):
    # Load both confusion matrices
    _cm_100_img = mpimg.imread(str(RUN_100_DIR / "confusion_matrix_normalized.png"))
    _cm_500_img = mpimg.imread(str(RUN_500_DIR / "confusion_matrix_normalized.png"))

    _fig, _axes = plt.subplots(1, 3, figsize=(24, 8))

    _axes[0].imshow(_cm_100_img)
    _axes[0].set_title('Confusion Matrix: 100 Epochs', fontsize=14, fontweight='bold')
    _axes[0].axis('off')

    _axes[1].imshow(_cm_500_img)
    _axes[1].set_title('Confusion Matrix: 500 Epochs', fontsize=14, fontweight='bold')
    _axes[1].axis('off')

    # Add annotation showing key improvements
    _axes[2].text(0.1, 0.9, "KEY IMPROVEMENTS (100→500):", fontsize=14, fontweight='bold', transform=_axes[2].transAxes)
    _axes[2].text(0.1, 0.8, "From confusion matrix analysis:", fontsize=12, transform=_axes[2].transAxes)
    _axes[2].text(0.1, 0.7, "• Hardhat: 0.75 → 0.85 (+13%)", fontsize=11, transform=_axes[2].transAxes, color='green')
    _axes[2].text(0.1, 0.65, "• NO-Hardhat: 0.55 → 0.65 (+18%)", fontsize=11, transform=_axes[2].transAxes, color='green')
    _axes[2].text(0.1, 0.60, "• NO-Mask: 0.58 → 0.68 (+17%)", fontsize=11, transform=_axes[2].transAxes, color='green')
    _axes[2].text(0.1, 0.55, "• Person: 0.73 → 0.81 (+11%)", fontsize=11, transform=_axes[2].transAxes, color='green')
    _axes[2].text(0.1, 0.50, "• Vehicle: 0.52 → 0.64 (+23% BEST)", fontsize=11, transform=_axes[2].transAxes, color='darkgreen', weight='bold')

    _axes[2].text(0.1, 0.4, "PROBLEM CLASSES (still struggling):", fontsize=12, fontweight='bold', transform=_axes[2].transAxes)
    _axes[2].text(0.1, 0.32, "• NO-Hardhat: 32% confused with background", fontsize=11, transform=_axes[2].transAxes, color='red')
    _axes[2].text(0.1, 0.27, "• NO-Mask: 32% confused with background", fontsize=11, transform=_axes[2].transAxes, color='red')
    _axes[2].text(0.1, 0.22, "• Vehicle: 36% confused with background", fontsize=11, transform=_axes[2].transAxes, color='red')

    _axes[2].text(0.1, 0.12, "RECOMMENDATION:", fontsize=12, fontweight='bold', transform=_axes[2].transAxes)
    _axes[2].text(0.1, 0.05, "Augment dataset with more examples of\nNO-Hardhat, NO-Mask, and vehicle classes", fontsize=10, transform=_axes[2].transAxes)

    _axes[2].axis('off')

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 16. Hyperparameter Experiments Configuration

    Define experiments to run for Section 8 of the report.
    """
    )
    return


@app.cell
def _(Path):
    # Define hyperparameter experiments
    EXPERIMENTS = {
        "baseline_100": {
            "name": "ppe_100",
            "description": "Baseline 100 epochs (already exists)",
            "params": {"epochs": 100, "lr0": 0.01, "batch": 16, "imgsz": 640, "dropout": 0.0, "mosaic": 1.0},
            "skip": True
        },
        "lr_low": {
            "name": "ppe_100_lr_0.001",
            "description": "Low learning rate",
            "params": {"epochs": 100, "lr0": 0.001, "batch": 16, "imgsz": 640, "dropout": 0.0, "mosaic": 1.0},
            "skip": True
        },
        "lr_high": {
            "name": "ppe_100_lr_0.02",
            "description": "High learning rate",
            "params": {"epochs": 100, "lr0": 0.02, "batch": 16, "imgsz": 640, "dropout": 0.0, "mosaic": 1.0},
            "skip": True
        },
        "batch_small": {
            "name": "ppe_100_batch_8",
            "description": "Smaller batch size (already exists)",
            "params": {"epochs": 100, "lr0": 0.01, "batch": 8, "imgsz": 640, "dropout": 0.0, "mosaic": 1.0},
            "skip": True
        },
        "batch_large": {
            "name": "ppe_100_batch_32",
            "description": "Larger batch size (fail, too big)",
            "params": {"epochs": 100, "lr0": 0.01, "batch": 32, "imgsz": 640, "dropout": 0.0, "mosaic": 1.0},
            "skip": True
        },
        "no_augment": {
            "name": "ppe_100_no_mosaic",
            "description": "No mosaic augmentation",
            "params": {"epochs": 100, "lr0": 0.01, "batch": 16, "imgsz": 640, "dropout": 0.0, "mosaic": 0.0},
            "skip": False
        },
        "dropout": {
            "name": "ppe_100_dropout_0.2",
            "description": "With dropout regularization",
            "params": {"epochs": 100, "lr0": 0.01, "batch": 16, "imgsz": 640, "dropout": 0.2, "mosaic": 1.0},
            "skip": False
        },
    }

    DATASET_YAML = Path.cwd() / "data" / "archive" / "css-data" / "data.yaml"
    PRETRAINED_MODEL = "yolov8n.pt"

    return DATASET_YAML, EXPERIMENTS, PRETRAINED_MODEL


@app.cell
def _(mo):
    mo.md(
        """
    ## 17. Run Hyperparameter Experiments

    """
    )
    return


@app.cell
def _():
    from loguru import logger
    logger.add("logs.log", rotation="500 MB") 
    from datetime import datetime
    import torch
    from ultralytics import YOLO
    print(datetime.now())
    return YOLO, datetime, logger, torch


@app.cell
def _(mo):
    train_experiments_btn = mo.ui.run_button(label="Train All Experiments", kind="danger")
    train_experiments_btn
    return (train_experiments_btn,)


@app.cell(hide_code=True)
def _(
    DATASET_YAML,
    EXPERIMENTS,
    PRETRAINED_MODEL,
    Path,
    YOLO,
    datetime,
    json,
    logger,
    pl,
    torch,
    train_experiments_btn,
):
    if train_experiments_btn.value:
        logger.info("=" * 80)
        logger.info("STARTING HYPERPARAMETER EXPERIMENTS")
        logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)

        _device = 0 if torch.cuda.is_available() else "cpu"
        logger.info(f"Device: {_device}")

        _experiment_results = {}

        for _exp_id, _exp in EXPERIMENTS.items():
            if _exp["skip"]:
                logger.info(f"⏭ Skipping {_exp_id} (already exists)")
                continue

            logger.info("-" * 80)
            logger.info(f"EXPERIMENT: {_exp_id}")
            logger.info(f"Description: {_exp['description']}")
            logger.info(f"Parameters: {_exp['params']}")

            _start_time = datetime.now()

            try:
                _model = YOLO(PRETRAINED_MODEL)
                logger.info(f"Training started: {_start_time.strftime('%H:%M:%S')}")

                _results = _model.train(
                    data=str(DATASET_YAML),
                    name=_exp["name"],
                    project="runs/saved",
                    exist_ok=True,
                    device=_device,
                    **_exp["params"]
                )

                _end_time = datetime.now()
                _duration = (_end_time - _start_time).total_seconds() / 3600

                logger.info(f"Training completed: {_end_time.strftime('%H:%M:%S')}")
                logger.info(f"Duration: {_duration:.2f} hours")

                _run_dir = Path("runs/saved") / _exp["name"]
                _results_csv = pl.read_csv(_run_dir / "results.csv")
                _final_metrics = _results_csv.tail(1)

                _mAP50 = _final_metrics["metrics/mAP50(B)"][0]
                _mAP5095 = _final_metrics["metrics/mAP50-95(B)"][0]
                _precision = _final_metrics["metrics/precision(B)"][0]
                _recall = _final_metrics["metrics/recall(B)"][0]

                logger.info(f"Final mAP50: {_mAP50:.4f}")
                logger.info(f"Final mAP50-95: {_mAP5095:.4f}")
                logger.info(f"Final Precision: {_precision:.4f}")
                logger.info(f"Final Recall: {_recall:.4f}")

                _summary = {
                    "experiment_id": _exp_id,
                    "run_name": _exp["name"],
                    "description": _exp["description"],
                    "parameters": _exp["params"],
                    "duration_hours": _duration,
                    "final_metrics": {
                        "mAP50": float(_mAP50),
                        "mAP50-95": float(_mAP5095),
                        "precision": float(_precision),
                        "recall": float(_recall),
                    },
                    "training_completed": _end_time.strftime('%Y-%m-%d %H:%M:%S')
                }

                with open(_run_dir / "summary.json", "w") as _f:
                    json.dump(_summary, _f, indent=2)

                _text_summary = f"""
    RUN: {_exp['name']}
    EXPERIMENT: {_exp_id} - {_exp['description']}
    COMPLETED: {_end_time.strftime('%Y-%m-%d %H:%M:%S')}
    DURATION: {_duration:.2f} hours

    HYPERPARAMETERS:
      Learning Rate (lr0): {_exp['params']['lr0']}
      Epochs: {_exp['params']['epochs']}
      Batch Size: {_exp['params']['batch']}
      Image Size: {_exp['params']['imgsz']}
      Dropout: {_exp['params']['dropout']}
      Mosaic Augmentation: {_exp['params']['mosaic']}

    FINAL METRICS:
      mAP@0.5: {_mAP50:.4f}
      mAP@0.5:0.95: {_mAP5095:.4f}
      Precision: {_precision:.4f}
      Recall: {_recall:.4f}
      F1 Score: {2 * _precision * _recall / (_precision + _recall):.4f}

    STATUS: ✓ SUCCESS
    """
                with open(_run_dir / "summary.txt", "w") as _f2:
                    _f2.write(_text_summary)

                logger.info("✓ Summary saved")
                _experiment_results[_exp_id] = _summary

            except Exception as e:
                logger.error(f"✗ FAILED: {_exp_id}")
                logger.error(f"Error: {str(e)}")
                _experiment_results[_exp_id] = {"status": "failed", "error": str(e)}

        logger.info("=" * 80)
        logger.info("ALL EXPERIMENTS COMPLETED")
        logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)

        print("\n✓ Training complete! Check experiment_log.txt for details.")
        print(f"Results saved in runs/saved/")
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 18. Multi-Run Comparison Dashboard

    Compare all hyperparameter experiments side-by-side.
    """
    )
    return


if __name__ == "__main__":
    app.run()
