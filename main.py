import marimo

__generated_with = "0.17.2"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Imports""")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import random
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import cv2
    from pathlib import Path
    import numpy as np
    from ultralytics import YOLO  # type: ignore
    from collections import Counter, defaultdict
    from bokeh.plotting import figure, show, output_notebook
    from bokeh.layouts import column
    from PIL import Image
    import torch
    from prettytable import PrettyTable
    from prettytable import TableStyle
    return (
        Counter,
        Path,
        PrettyTable,
        TableStyle,
        YOLO,
        cv2,
        defaultdict,
        mpimg,
        np,
        plt,
        random,
        torch,
    )


@app.cell
def _(TableStyle):
    TABLES_STYLE = TableStyle.MARKDOWN  # DEFAULT | MARKDOWN | ASCII
    return (TABLES_STYLE,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # CHECKS & SETTINGS
    ## NOTE: CHECK CONSTANTS BELOW TO ENSURE CORRECTNESS BEFORE RUNNING THE NOTEBOOK.

    ** Navigation**: [Jump to Train Button](#training-configuration)
    """
    )
    return


@app.cell
def _(mo):
    regenerate_yaml = mo.ui.checkbox(
        label="# Regenerate YAML file (uncheck to skip if file is correct)",
        value=False,
    )
    regenerate_yaml
    return (regenerate_yaml,)


@app.cell(hide_code=True)
def _():
    def ascii_table_to_latex(ascii_table, caption="", label="", precision=3):
        """
        tmp for latex stuff, dont forget to change table style to default or ascii
        """
        lines = ascii_table.strip().split("\n")

        data_lines = [line for line in lines if not line.startswith("+")]

        if not data_lines:
            return ""
        header_line = data_lines[0]
        data_rows = data_lines[1:]

        header_cells = [cell.strip() for cell in header_line.split("|")[1:-1]]
        header_cells = [cell.replace("_", "\\_") for cell in header_cells]

        parsed_rows = []
        for row in data_rows:
            cells = [cell.strip() for cell in row.split("|")[1:-1]]
            formatted_cells = []
            for cell in cells:
                try:
                    num = float(cell)
                    if num.is_integer():
                        formatted_cells.append(str(int(num)))
                    else:
                        formatted_cells.append(f"{num:.{precision}f}")
                except ValueError:
                    formatted_cells.append(cell.replace("_", "\\_"))
            parsed_rows.append(formatted_cells)

        num_cols = len(header_cells)
        colspec = "{" + "r" * num_cols + "}"

        latex_code = []

        if caption or label:
            latex_code.append("\\begin{table}[h]")
            if caption:
                latex_code.append(f"    \\caption{{{caption}}}")
            if label:
                latex_code.append(f"    \\label{{{label}}}")
            latex_code.append("    \\centering")

        latex_code.append(f"    \\begin{{tabular}}{colspec}")
        latex_code.append("    \\toprule")

        header_str = " & ".join(header_cells)
        latex_code.append(f"    {header_str} \\\\")
        latex_code.append("    \\midrule")

        for row in parsed_rows:
            row_str = " & ".join(row)
            latex_code.append(f"    {row_str} \\\\")

        latex_code.append("    \\bottomrule")
        latex_code.append("    \\end{tabular}")

        if caption or label:
            latex_code.append("\\end{table}")

        return "\n".join(latex_code)


    ascii_table = """
    """

    latex_output = ascii_table_to_latex(
        ascii_table, caption="", label="", precision=3
    )
    print(latex_output)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # CS4287-CNN: Construction Safety Equipment Detection

    **Authors**: MYKOLA VASKEVYCH (22372199), Teammate Name (ID2)

    **Status**: Code executes to completion: YES

    ## Overview
    The model identifies safety equipment (hardhats, masks, safety vests) and flags violations when workers lack proper protection.
    The model uses yolo
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Dataset Configuration

    Generate YOLO-compatible data.yaml configuration file and verify dataset structure.
    """
    )
    return


@app.cell
def _(
    CLASS_NAMES,
    DATASET_ROOT,
    TEST_IMAGES_PATH,
    TRAINING_IMAGES_PATH,
    VALIDATION_IMAGES_PATH,
    YAML_CONFIG_PATH,
    regenerate_yaml,
):
    # Generate YAML using CLASS_NAMES constant to ensure consistency
    _names_yaml = "\n  ".join([f"{k}: {v}" for k, v in CLASS_NAMES.items()])

    _yaml_content = f"""path: {DATASET_ROOT.resolve()}
    train: train/images
    val: valid/images
    test: test/images

    nc: {len(CLASS_NAMES)}
    names:
      {_names_yaml}
    """

    if regenerate_yaml.value:
        with open(YAML_CONFIG_PATH, "w") as _f:
            _f.write(_yaml_content)
        print(" Generated data.yaml:")
        print(_yaml_content)
    else:
        print(" Skipped YAML generation (using existing file)")

    print("\nVerifying paths:")
    print(f"Train exists: {TRAINING_IMAGES_PATH.exists()}")
    print(f"Val exists: {VALIDATION_IMAGES_PATH.exists()}")
    print(f"Test exists: {TEST_IMAGES_PATH.exists()}")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Dataset Structure Analysis

    Examine the distribution of images and labels across train/validation/test splits.
    """
    )
    return


@app.cell
def _(np, random, torch):
    RANDOM_SEED = 42
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    return (RANDOM_SEED,)


@app.cell
def _(RANDOM_SEED, torch):
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)


    if torch.cuda.is_available():
        DEVICE = 0
        _device_name = torch.cuda.get_device_name(0)
        _vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    else:
        DEVICE = "cpu"
    return (DEVICE,)


@app.cell
def _(Path):
    # Dataset paths
    DATASET_ROOT = Path.cwd() / "data" / "archive" / "css-data"
    TRAINING_IMAGES_PATH = (DATASET_ROOT / "train" / "images").resolve()
    TRAINING_LABELS_PATH = (DATASET_ROOT / "train" / "labels").resolve()
    VALIDATION_IMAGES_PATH = (DATASET_ROOT / "valid" / "images").resolve()
    VALIDATION_LABELS_PATH = (DATASET_ROOT / "valid" / "labels").resolve()
    TEST_IMAGES_PATH = (DATASET_ROOT / "test" / "images").resolve()
    TEST_LABELS_PATH = (DATASET_ROOT / "test" / "labels").resolve()
    YAML_CONFIG_PATH = DATASET_ROOT / "data.yaml"

    # Model paths
    PRETRAINED_MODEL_PATH = "yolov8n.pt"
    TRAINING_OUTPUT_DIR = "runs/train"
    TRAINING_RUN_NAME = "ppe_detection"
    TRAINED_MODEL_PATH = (
        Path(TRAINING_OUTPUT_DIR) / TRAINING_RUN_NAME / "weights" / "best.pt"
    )
    return (
        DATASET_ROOT,
        PRETRAINED_MODEL_PATH,
        TEST_IMAGES_PATH,
        TRAINED_MODEL_PATH,
        TRAINING_IMAGES_PATH,
        TRAINING_LABELS_PATH,
        TRAINING_OUTPUT_DIR,
        TRAINING_RUN_NAME,
        VALIDATION_IMAGES_PATH,
        YAML_CONFIG_PATH,
    )


@app.cell
def _():
    # Training parameters
    EPOCHS = 500  # Reduce to 10 for quick testing or CPU training
    IMAGE_SIZE = 640  # YOLO standard input size
    BATCH_SIZE = 16  # Reduce to 4-8 for low VRAM or CPU

    # Detection parameters
    CONFIDENCE_THRESHOLD = 0.25  # Minimum confidence for detections (0.0-1.0)
    return BATCH_SIZE, CONFIDENCE_THRESHOLD, EPOCHS, IMAGE_SIZE


@app.cell
def _():
    # Visualization parameters
    NUM_SAMPLE_IMAGES = 6
    NUM_COMPARISON_IMAGES = 4
    NUM_BASELINE_TEST_SAMPLES = 3
    return NUM_BASELINE_TEST_SAMPLES, NUM_COMPARISON_IMAGES, NUM_SAMPLE_IMAGES


@app.cell
def _(
    BATCH_SIZE,
    CONFIDENCE_THRESHOLD,
    DEVICE,
    EPOCHS,
    IMAGE_SIZE,
    PrettyTable,
    TABLES_STYLE,
    mo,
):
    # Display configuration summary
    _table = PrettyTable()
    _table.field_names = ["Parameter", "Value", "Description"]
    _table.add_rows(
        [
            ["Device", f"`{DEVICE}`", "Training device (0=GPU, 'cpu'=CPU)"],
            ["Epochs", f"`{EPOCHS}`", "Training iterations through dataset"],
            ["Image Size", f"`{IMAGE_SIZE}px`", "Input resolution"],
            ["Batch Size", f"`{BATCH_SIZE}`", "Images per training step"],
            [
                "Confidence",
                f"`{CONFIDENCE_THRESHOLD}`",
                "Min score for detections",
            ],
        ]
    )
    _table.set_style(TABLES_STYLE)
    mo.md(_table.get_string())
    return


@app.cell
def _(DATASET_ROOT, PrettyTable, TABLES_STYLE, mo):
    # Display dataset structure summary
    _table = PrettyTable()
    _table.field_names = ["Split", "Images", "Labels"]
    _table.add_rows(
        [
            [
                "TRAIN",
                len(list((DATASET_ROOT / "train" / "images").glob("*.jpg"))),
                len(list((DATASET_ROOT / "train" / "labels").glob("*.txt"))),
            ],
            [
                "VALID",
                len(list((DATASET_ROOT / "valid" / "images").glob("*.jpg"))),
                len(list((DATASET_ROOT / "valid" / "labels").glob("*.txt"))),
            ],
            [
                "TEST",
                len(list((DATASET_ROOT / "test" / "images").glob("*.jpg"))),
                len(list((DATASET_ROOT / "test" / "labels").glob("*.txt"))),
            ],
        ]
    )
    _table.set_style(TABLES_STYLE)
    mo.md(_table.get_string())
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Label Format Inspection

    YOLO format uses: `class_id x_center y_center width height` (normalized 0-1).
    """
    )
    return


@app.cell
def _(PrettyTable, TABLES_STYLE, TRAINING_LABELS_PATH, mo):
    # Display sample label file
    _label_files = list(TRAINING_LABELS_PATH.glob("*.txt"))
    _first_label = _label_files[0]

    with open(_first_label, "r") as _f:
        _lines = _f.readlines()[:10]

    _table = PrettyTable()
    _table.field_names = [
        "Line",
        "class_id",
        "x_center",
        "y_center",
        "width",
        "height",
    ]
    _table.add_rows(
        [[_i] + _line.strip().split() for _i, _line in enumerate(_lines, 1)]
    )
    _table.set_style(TABLES_STYLE)

    mo.md(f"""
    ## SAMPLE LABEL FILE

    **Label file:** `{_first_label.name}`

    {_table.get_string()}

    **Total objects in this image:** {len(_lines)}
    """)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Class Distribution Analysis

    Count objects per class across all dataset splits to identify class imbalance.
    """
    )
    return


@app.cell
def _(Counter, DATASET_ROOT, PrettyTable, TableStyle, mo):
    _all_class_ids = []
    for _split in ["train", "valid", "test"]:
        _label_path = DATASET_ROOT / _split / "labels"
        for _label_file in _label_path.glob("*.txt"):
            with open(_label_file, "r") as _f:
                for _line in _f:
                    _parts = _line.strip().split()
                    if _parts:
                        _class_id = int(_parts[0])
                        _all_class_ids.append(_class_id)

    _class_counts = Counter(_all_class_ids)

    _table = PrettyTable()
    _table.field_names = ["Class ID", "Count", "Percentage"]
    _table.add_rows(
        [
            [
                _class_id,
                _class_counts[_class_id],
                f"{(_class_counts[_class_id] / len(_all_class_ids)) * 100:.2f}%",
            ]
            for _class_id in sorted(_class_counts.keys())
        ]
    )
    _table.set_style(TableStyle.MARKDOWN)

    mo.md(f"""
    ## CLASS DISTRIBUTION

    **Total objects:** {len(_all_class_ids)}  
    **Number of unique classes:** {len(_class_counts)}

    {_table.get_string()}
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""# Actual Code""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Sample Images Visualization""")
    return


@app.cell
def _(cv2):
    def draw_boxes_on_image(img_path, label_path, class_names, colors):
        """Draw YOLO bounding boxes on image"""
        _img = cv2.imread(str(img_path))
        _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
        _h, _w = _img.shape[:2]

        with open(label_path, "r") as _f:
            for _line in _f:
                _parts = _line.strip().split()
                if not _parts:
                    continue

                _class_id = int(_parts[0])
                _x_center = float(_parts[1])
                _y_center = float(_parts[2])
                _width = float(_parts[3])
                _height = float(_parts[4])

                # Convert YOLO format to pixel coordinates
                _x1 = int((_x_center - _width / 2) * _w)
                _y1 = int((_y_center - _height / 2) * _h)
                _x2 = int((_x_center + _width / 2) * _w)
                _y2 = int((_y_center + _height / 2) * _h)

                # Draw rectangle
                _color = colors.get(_class_id, (255, 255, 255))
                cv2.rectangle(_img, (_x1, _y1), (_x2, _y2), _color, 2)

                # Add label
                _label = class_names.get(_class_id, f"Class {_class_id}")
                _label_size, _baseline = cv2.getTextSize(
                    _label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                _y1_label = max(_y1, _label_size[1] + 10)
                cv2.rectangle(
                    _img,
                    (_x1, _y1_label - _label_size[1] - 10),
                    (_x1 + _label_size[0], _y1_label + _baseline - 10),
                    _color,
                    cv2.FILLED,
                )
                cv2.putText(
                    _img,
                    _label,
                    (_x1, _y1_label - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2,
                )

        return _img
    return (draw_boxes_on_image,)


@app.cell
def _(
    NUM_SAMPLE_IMAGES,
    TRAINING_IMAGES_PATH,
    TRAINING_LABELS_PATH,
    draw_boxes_on_image,
    plt,
):
    # Bounding box colors (BGR format for OpenCV)
    BBOX_COLORS = {
        0: (0, 255, 0),  # Hardhat    - Green
        1: (255, 255, 0),  # Mask       - Cyan
        2: (0, 0, 255),  # NO-Hardhat - Red
        3: (0, 0, 255),  # NO-Mask    - Red
        4: (0, 0, 255),  # NO-Safety Vest - Red
        5: (255, 0, 255),  # Person          - Magenta
        6: (0, 165, 255),  # Safety Cone     - Orange
        7: (0, 255, 0),  # Safety Vest     - Green
        8: (128, 128, 128),  # machinery       - Gray
        9: (255, 0, 0),  # vehicle         - Blue
    }

    # Class definitions
    CLASS_NAMES = {
        0: "Hardhat",
        1: "Mask",
        2: "NO-Hardhat",
        3: "NO-Mask",
        4: "NO-Safety Vest",
        5: "Person",
        6: "Safety Cone",
        7: "Safety Vest",
        8: "machinery",
        9: "vehicle",
    }


    print("=" * 50)
    print("VISUALIZING SAMPLE IMAGES")
    print("=" * 50)

    _image_files = list(TRAINING_IMAGES_PATH.glob("*.jpg"))[:NUM_SAMPLE_IMAGES]
    _fig, _axes = plt.subplots(2, 3, figsize=(15, 10))
    _axes = _axes.flatten()

    for _idx, _img_file in enumerate(_image_files):
        _label_file = TRAINING_LABELS_PATH / (_img_file.stem + ".txt")

        if _label_file.exists():
            _img_with_boxes = draw_boxes_on_image(
                _img_file, _label_file, CLASS_NAMES, BBOX_COLORS
            )
            _axes[_idx].imshow(_img_with_boxes)
            _axes[_idx].set_title(f"Image: {_img_file.name}", fontsize=10)
            _axes[_idx].axis("off")

    plt.tight_layout()
    plt.show()

    print("\nLegend:")
    print("  Green: Hardhat, Safety Vest (PPE worn correctly)")
    print("  Red: NO-Hardhat, NO-Mask, NO-Safety Vest (violations)")
    print("  Magenta: Person")
    print("  Orange: Safety Cone")
    print("  Gray: Machinery")
    print("  Blue: Vehicle")
    return (CLASS_NAMES,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Data Quality Notes

    Some images have labeling ambiguities: misclassified clothing items or crowded scenes with occlusions.
    """
    )
    return


@app.cell
def _(TRAINING_LABELS_PATH, defaultdict):
    print("=" * 50)
    print("DATA QUALITY OBSERVATIONS")
    print("=" * 50)

    _cooccurrence = defaultdict(int)

    for _label_file in TRAINING_LABELS_PATH.glob("*.txt"):
        _classes_in_image = set()
        with open(_label_file, "r") as _f:
            for _line in _f:
                _parts = _line.strip().split()
                if _parts:
                    _classes_in_image.add(int(_parts[0]))

        # Check for suspicious combinations (Person with contradictory PPE states)
        if 5 in _classes_in_image:
            if 7 in _classes_in_image and 4 in _classes_in_image:
                _cooccurrence["Person with BOTH vest AND no-vest"] += 1
            if 0 in _classes_in_image and 2 in _classes_in_image:
                _cooccurrence["Person with BOTH hardhat AND no-hardhat"] += 1

    print("\nPotential labeling inconsistencies found:")
    for _issue, _count in _cooccurrence.items():
        print(f"  {_issue}: {_count} images")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Model Training

    Load pre-trained YOLOv8 model and fine-tune on PPE detection dataset.
    """
    )
    return


@app.cell
def _(PRETRAINED_MODEL_PATH, YOLO):
    # Load YOLOv8 nano model pre-trained on COCO dataset
    pretrained_model = YOLO(PRETRAINED_MODEL_PATH)
    pretrained_model.info()
    return (pretrained_model,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Baseline Performance

    Test pre-trained COCO model on construction site images to establish baseline detection capabilities.
    """
    )
    return


@app.cell
def _(
    CONFIDENCE_THRESHOLD,
    NUM_BASELINE_TEST_SAMPLES,
    TEST_IMAGES_PATH,
    pretrained_model,
):
    pretrained_test_images = list(TEST_IMAGES_PATH.glob("*.jpg"))[
        :NUM_BASELINE_TEST_SAMPLES
    ]

    print("Testing pre-trained COCO model on TEST set (unbiased baseline):")
    for _img_file in pretrained_test_images:
        _results = pretrained_model.predict(
            source=str(_img_file),
            conf=CONFIDENCE_THRESHOLD,
            save=False,
            verbose=True,
        )
        _result = _results[0]
        print(f"{_img_file.name}: {len(_result.boxes)} objects detected")
    return (pretrained_test_images,)


@app.cell
def _(
    CONFIDENCE_THRESHOLD,
    cv2,
    plt,
    pretrained_model,
    pretrained_test_images,
):
    _fig, _axes = plt.subplots(1, 3, figsize=(15, 5))

    for _idx, _img_file in enumerate(pretrained_test_images):
        _results = pretrained_model.predict(
            source=str(_img_file),
            conf=CONFIDENCE_THRESHOLD,
            save=False,
            verbose=False,
        )

        _result = _results[0]
        _plotted_img = _result.plot()
        _plotted_img_rgb = cv2.cvtColor(_plotted_img, cv2.COLOR_BGR2RGB)

        _axes[_idx].imshow(_plotted_img_rgb)
        _axes[_idx].set_title(f"{_img_file.name}\n{len(_result.boxes)} detections")
        _axes[_idx].axis("off")

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    train_button = mo.ui.run_button(label="Train Model")
    train_button
    return (train_button,)


@app.cell
def _(
    BATCH_SIZE,
    DEVICE,
    EPOCHS,
    IMAGE_SIZE,
    TRAINING_OUTPUT_DIR,
    TRAINING_RUN_NAME,
    YAML_CONFIG_PATH,
    mo,
    pretrained_model,
    train_button,
):
    mo.stop(
        not train_button.value,
        mo.md("Press 'Train Model' button to start training"),
    )

    print(f"Training with parameters:")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Image Size: {IMAGE_SIZE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Device: {DEVICE}")

    training_results = pretrained_model.train(
        data=str(YAML_CONFIG_PATH),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=TRAINING_OUTPUT_DIR,
        name=TRAINING_RUN_NAME,
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Training Results Analysis

    Load trained model and visualize training metrics, confusion matrix, and validation predictions.
    """
    )
    return


@app.cell
def _(TRAINED_MODEL_PATH, YOLO, mo):
    mo.stop(
        not TRAINED_MODEL_PATH.exists(),
        mo.md(
            f"**Trained model not found at `{TRAINED_MODEL_PATH}`. Please train the model first.**"
        ),
    )

    trained_model = YOLO(TRAINED_MODEL_PATH)
    print(f"Loaded trained model from: {TRAINED_MODEL_PATH}")
    return (trained_model,)


@app.cell
def _(TRAINED_MODEL_PATH, mo, mpimg, plt):
    mo.stop(
        not TRAINED_MODEL_PATH.exists(),
        mo.md("**Train the model first to see results.**"),
    )

    _results_dir = TRAINED_MODEL_PATH.parent.parent

    _fig, _axes = plt.subplots(2, 2, figsize=(15, 12))

    _images_to_show = [
        ("results.png", "Training Metrics"),
        ("confusion_matrix_normalized.png", "Confusion Matrix"),
        ("val_batch0_labels.jpg", "Validation: Ground Truth"),
        ("val_batch0_pred.jpg", "Validation: Predictions"),
    ]

    for _idx, (_img_name, _title) in enumerate(_images_to_show):
        _img_path = _results_dir / _img_name
        if _img_path.exists():
            _img = mpimg.imread(str(_img_path))
            _axes[_idx // 2, _idx % 2].imshow(_img)
            _axes[_idx // 2, _idx % 2].set_title(_title)
            _axes[_idx // 2, _idx % 2].axis("off")

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Detailed Training Metrics

    Large-format visualizations for detailed analysis of training performance.
    """
    )
    return


@app.cell
def _(TRAINED_MODEL_PATH, mo, mpimg, plt):
    mo.stop(
        not TRAINED_MODEL_PATH.exists(),
        mo.md("**Train the model first to see detailed results.**"),
    )

    _results_dir = TRAINED_MODEL_PATH.parent.parent

    # Training Metrics
    _fig1 = plt.figure(figsize=(20, 12))
    _img1 = mpimg.imread(str(_results_dir / "results.png"))
    plt.imshow(_img1)
    plt.title("Training Metrics (Loss, Precision, Recall, mAP)", fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # Confusion Matrix
    _fig2 = plt.figure(figsize=(16, 16))
    _img2 = mpimg.imread(str(_results_dir / "confusion_matrix_normalized.png"))
    plt.imshow(_img2)
    plt.title("Confusion Matrix (Normalized)", fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # Validation Comparison
    _fig3, _axes3 = plt.subplots(1, 2, figsize=(24, 12))
    _img_labels = mpimg.imread(str(_results_dir / "val_batch0_labels.jpg"))
    _img_preds = mpimg.imread(str(_results_dir / "val_batch0_pred.jpg"))

    _axes3[0].imshow(_img_labels)
    _axes3[0].set_title("Validation: Ground Truth Labels", fontsize=14)
    _axes3[0].axis("off")

    _axes3[1].imshow(_img_preds)
    _axes3[1].set_title("Validation: Model Predictions", fontsize=14)
    _axes3[1].axis("off")

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Model Comparison: Pre-trained vs Fine-tuned

    Side-by-side comparison of COCO pre-trained model vs PPE fine-tuned model on test images.
    """
    )
    return


@app.cell
def _(
    CONFIDENCE_THRESHOLD,
    NUM_COMPARISON_IMAGES,
    TEST_IMAGES_PATH,
    TRAINED_MODEL_PATH,
    cv2,
    mo,
    plt,
    pretrained_model,
    trained_model,
):
    mo.stop(
        not TRAINED_MODEL_PATH.exists(),
        mo.md("**Train the model first to see comparison.**"),
    )

    _comparison_images = list(TEST_IMAGES_PATH.glob("*.jpg"))[
        :NUM_COMPARISON_IMAGES
    ]

    _fig, _axes = plt.subplots(2, 4, figsize=(20, 10))

    for _idx, _img_file in enumerate(_comparison_images):
        # Pre-trained COCO model
        _results_pretrained = pretrained_model.predict(
            str(_img_file), conf=CONFIDENCE_THRESHOLD, verbose=False
        )
        _img_pretrained = _results_pretrained[0].plot()
        _img_pretrained_rgb = cv2.cvtColor(_img_pretrained, cv2.COLOR_BGR2RGB)

        # Fine-tuned PPE model
        _results_finetuned = trained_model.predict(
            str(_img_file), conf=CONFIDENCE_THRESHOLD, verbose=False
        )
        _img_finetuned = _results_finetuned[0].plot()
        _img_finetuned_rgb = cv2.cvtColor(_img_finetuned, cv2.COLOR_BGR2RGB)

        # Display
        _axes[0, _idx].imshow(_img_pretrained_rgb)
        _axes[0, _idx].set_title(
            f"COCO: {len(_results_pretrained[0].boxes)} detections", fontsize=10
        )
        _axes[0, _idx].axis("off")

        _axes[1, _idx].imshow(_img_finetuned_rgb)
        _axes[1, _idx].set_title(
            f"PPE: {len(_results_finetuned[0].boxes)} detections", fontsize=10
        )
        _axes[1, _idx].axis("off")

    _fig.text(
        0.02,
        0.75,
        "Pre-trained\n(COCO)",
        fontsize=14,
        weight="bold",
        va="center",
        rotation=90,
    )
    _fig.text(
        0.02,
        0.25,
        "Fine-tuned\n(PPE)",
        fontsize=14,
        weight="bold",
        va="center",
        rotation=90,
    )

    plt.tight_layout()
    plt.gca()
    return


if __name__ == "__main__":
    app.run()
