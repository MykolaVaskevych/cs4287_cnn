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
    import torch
    from prettytable import PrettyTable
    from prettytable import TableStyle
    import polars as pl
    import yaml
    import seaborn as sns
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
        pl,
        plt,
        random,
        sns,
        torch,
        yaml,
    )


@app.cell
def _(TableStyle):
    TABLES_STYLE = TableStyle.MARKDOWN  # DEFAULT | MARKDOWN | ASCII
    return (TABLES_STYLE,)


@app.cell
def _(mo):
    mo.md(r"""# CHECKS & SETTINGS""")
    return


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
    return


@app.cell
def _(Path):
    DATASET_ROOT = Path.cwd() / "data" / "archive" / "css-data"
    TRAINING_IMAGES_PATH = (DATASET_ROOT / "train" / "images").resolve()
    TRAINING_LABELS_PATH = (DATASET_ROOT / "train" / "labels").resolve()
    TEST_IMAGES_PATH = (DATASET_ROOT / "test" / "images").resolve()
    PRETRAINED_MODEL_PATH = "yolov8n.pt"
    return (
        DATASET_ROOT,
        PRETRAINED_MODEL_PATH,
        TEST_IMAGES_PATH,
        TRAINING_IMAGES_PATH,
        TRAINING_LABELS_PATH,
    )


@app.cell
def _():
    CONFIDENCE_THRESHOLD = 0.25
    NUM_SAMPLE_IMAGES = 6
    NUM_BASELINE_TEST_SAMPLES = 3
    return CONFIDENCE_THRESHOLD, NUM_BASELINE_TEST_SAMPLES, NUM_SAMPLE_IMAGES


@app.cell
def _(YOLO):
    def train_model(
        epochs,
        batch,
        lr0=0.01,
        dropout=0.0,
        mosaic=1.0,
        name="experiment",
        project="runs/saved_2",
        seed=0,
    ):
        model = YOLO("yolov8n.pt")
        results = model.train(
            data="data/archive/css-data/data.yaml",
            epochs=epochs,
            imgsz=640,
            batch=batch,
            device=0,
            project=project,
            name=name,
            exist_ok=False,
            lr0=lr0,
            dropout=dropout,
            mosaic=mosaic,
            patience=100,
            save=True,
            seed=seed,
            deterministic=True,
        )
        return results
    return (train_model,)


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
    print(len(list((DATASET_ROOT / "test" / "images").glob("*.jpg")))+len(list((DATASET_ROOT / "valid" / "images").glob("*.jpg")))+len(list((DATASET_ROOT / "train" / "images").glob("*.jpg"))))

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
    mo.md(
        r"""
    ## Data Validation & Cleaning

    Validate dataset integrity and remove corrupted/invalid data before training.
    """
    )
    return


@app.cell
def _(cv2):
    def validate_image(img_path):
        """Check if image can be loaded and is not corrupted"""
        try:
            _img = cv2.imread(str(img_path))
            if _img is None:
                return False, "Failed to load"
            if _img.size == 0:
                return False, "Empty image"
            return True, None
        except Exception as _e:
            return False, str(_e)


    def validate_label(label_path, num_classes=10):
        """Validate label file content"""
        _issues = []
        try:
            with open(label_path, "r") as _f:
                _lines = _f.readlines()

            if not _lines:
                return True, None  # Empty labels are valid (no objects)

            for _i, _line in enumerate(_lines, 1):
                _parts = _line.strip().split()
                if len(_parts) != 5:
                    _issues.append(
                        f"Line {_i}: Invalid format (expected 5 values)"
                    )
                    continue

                try:
                    _cls, _x, _y, _w, _h = map(float, _parts)

                    if not (0 <= _cls < num_classes):
                        _issues.append(f"Line {_i}: Invalid class {int(_cls)}")
                    if not (0 <= _x <= 1 and 0 <= _y <= 1):
                        _issues.append(f"Line {_i}: Center coords out of range")
                    if not (0 < _w <= 1 and 0 < _h <= 1):
                        _issues.append(f"Line {_i}: Invalid dimensions")
                except ValueError:
                    _issues.append(f"Line {_i}: Non-numeric values")

            return len(_issues) == 0, _issues if _issues else None
        except Exception as _e:
            return False, [str(_e)]
    return validate_image, validate_label


@app.cell
def _(DATASET_ROOT, validate_image, validate_label):
    def scan_dataset(dataset_root):
        """Scan all splits and collect validation issues"""
        _issues = {
            "corrupted_images": [],
            "missing_labels": [],
            "missing_images": [],
            "invalid_labels": [],
        }

        for _split in ["train", "valid", "test"]:
            _img_dir = dataset_root / _split / "images"
            _lbl_dir = dataset_root / _split / "labels"

            if not _img_dir.exists() or not _lbl_dir.exists():
                continue

            # Check images
            for _img_path in _img_dir.glob("*.jpg"):
                _lbl_path = _lbl_dir / f"{_img_path.stem}.txt"

                # Validate image
                _valid, _error = validate_image(_img_path)
                if not _valid:
                    _issues["corrupted_images"].append(
                        (_split, _img_path.name, _error)
                    )

                # Check label exists
                if not _lbl_path.exists():
                    _issues["missing_labels"].append((_split, _img_path.name))

            # Check for orphaned labels
            for _lbl_path in _lbl_dir.glob("*.txt"):
                _img_path = _img_dir / f"{_lbl_path.stem}.jpg"
                if not _img_path.exists():
                    _issues["missing_images"].append((_split, _lbl_path.name))
                else:
                    # Validate label content
                    _valid, _errors = validate_label(_lbl_path)
                    if not _valid:
                        _issues["invalid_labels"].append(
                            (_split, _lbl_path.name, _errors)
                        )

        return _issues


    validation_issues = scan_dataset(DATASET_ROOT)
    return (validation_issues,)


@app.cell
def _(PrettyTable, TABLES_STYLE, mo, validation_issues):
    _table = PrettyTable()
    _table.field_names = ["Issue Type", "Count"]
    _table.add_rows(
        [
            ["Corrupted Images", len(validation_issues["corrupted_images"])],
            ["Missing Labels", len(validation_issues["missing_labels"])],
            ["Orphaned Labels", len(validation_issues["missing_images"])],
            ["Invalid Labels", len(validation_issues["invalid_labels"])],
        ]
    )
    _table.set_style(TABLES_STYLE)

    _total = sum(len(_v) for _v in validation_issues.values())

    mo.md(f"""
    ## Validation Results

    {_table.get_string()}

    **Total Issues Found:** {_total}
    """)
    return


@app.cell
def _(mo, validation_issues):
    _has_issues = any(len(_v) > 0 for _v in validation_issues.values())

    if _has_issues:
        clean_btn = mo.ui.run_button(label="Clean Dataset")
        clean_btn
    else:
        clean_btn = None
        mo.md("**Dataset is clean - no issues found**")
    return (clean_btn,)


@app.cell
def _(DATASET_ROOT, Path, clean_btn, mo, validation_issues):
    mo.stop(
        clean_btn is None or not clean_btn.value,
        mo.md("Click button to clean dataset"),
    )

    _cleaned = {"images": 0, "labels": 0}

    # Remove corrupted images and their labels
    for _split, _img_name, _error in validation_issues["corrupted_images"]:
        _img_path = DATASET_ROOT / _split / "images" / _img_name
        _lbl_path = (
            DATASET_ROOT / _split / "labels" / f"{Path(_img_name).stem}.txt"
        )

        if _img_path.exists():
            _img_path.unlink()
            _cleaned["images"] += 1
        if _lbl_path.exists():
            _lbl_path.unlink()
            _cleaned["labels"] += 1

    # Remove images without labels
    for _split, _img_name in validation_issues["missing_labels"]:
        _img_path = DATASET_ROOT / _split / "images" / _img_name
        if _img_path.exists():
            _img_path.unlink()
            _cleaned["images"] += 1

    # Remove orphaned labels
    for _split, _lbl_name in validation_issues["missing_images"]:
        _lbl_path = DATASET_ROOT / _split / "labels" / _lbl_name
        if _lbl_path.exists():
            _lbl_path.unlink()
            _cleaned["labels"] += 1

    # Remove invalid labels (and corresponding images)
    for _split, _lbl_name, _errors in validation_issues["invalid_labels"]:
        _lbl_path = DATASET_ROOT / _split / "labels" / _lbl_name
        _img_path = (
            DATASET_ROOT / _split / "images" / f"{Path(_lbl_name).stem}.jpg"
        )

        if _lbl_path.exists():
            _lbl_path.unlink()
            _cleaned["labels"] += 1
        if _img_path.exists():
            _img_path.unlink()
            _cleaned["images"] += 1

    mo.md(f"""ok
    **Removed:** {_cleaned["images"]} images, {_cleaned["labels"]} labels

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
    return


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
    mo.md(
        r"""
    ## Training Experiments

    Run various experiments with different hyperparameters. Each experiment saves to runs/saved_2.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""### Baseline (100 epochs, batch=16, lr=0.01)""")
    return


@app.cell
def _(mo):
    train_baseline = mo.ui.run_button(label="Train Baseline", )
    train_baseline
    return (train_baseline,)


@app.cell
def _(mo, train_baseline, train_model):
    mo.stop(not train_baseline.value)
    train_model(epochs=100, batch=16, name="ppe_100", seed=42, exist_ok=False)
    return


@app.cell
def _(mo):
    mo.md(r"""### Batch Size 8""")
    return


@app.cell
def _(mo):
    train_batch8 = mo.ui.run_button(label="Train Batch=8")
    train_batch8
    return (train_batch8,)


@app.cell
def _(mo, train_batch8, train_model):
    mo.stop(not train_batch8.value)
    train_model(epochs=100, batch=8, name="ppe_100_batch_8", seed=43)
    return


@app.cell
def _(mo):
    mo.md(r"""### No Mosaic Augmentation""")
    return


@app.cell
def _(mo):
    train_no_mosaic = mo.ui.run_button(label="Train No Mosaic")
    train_no_mosaic
    return (train_no_mosaic,)


@app.cell
def _(mo, train_model, train_no_mosaic):
    mo.stop(not train_no_mosaic.value)
    train_model(
        epochs=100, batch=16, mosaic=0.0, name="ppe_100_no_mosaic", seed=44
    )
    return


@app.cell
def _(mo):
    mo.md(r"""### Learning Rate 0.001""")
    return


@app.cell
def _(mo):
    train_lr001 = mo.ui.run_button(label="Train LR=0.001")
    train_lr001
    return (train_lr001,)


@app.cell
def _(mo, train_lr001, train_model):
    mo.stop(not train_lr001.value)
    train_model(epochs=100, batch=16, lr0=0.001, name="ppe_100_lr_0.001", seed=45)
    return


@app.cell
def _(mo):
    mo.md(r"""### Learning Rate 0.02""")
    return


@app.cell
def _(mo):
    train_lr002 = mo.ui.run_button(label="Train LR=0.02")
    train_lr002
    return (train_lr002,)


@app.cell
def _(mo, train_lr002, train_model):
    mo.stop(not train_lr002.value)
    train_model(epochs=100, batch=16, lr0=0.02, name="ppe_100_lr_0.02", seed=46)
    return


@app.cell
def _(mo):
    mo.md(r"""### Dropout 0.2""")
    return


@app.cell
def _(mo):
    train_dropout = mo.ui.run_button(label="Train Dropout=0.2")
    train_dropout
    return (train_dropout,)


@app.cell
def _(mo, train_dropout, train_model):
    mo.stop(not train_dropout.value)
    train_model(
        epochs=100, batch=16, dropout=0.2, name="ppe_100_dropout_0.2", seed=47
    )
    return


@app.cell
def _(mo):
    mo.md(r"""### Combined Best (batch=8, no mosaic, 750 epochs)""")
    return


@app.cell
def _(mo):
    train_combined = mo.ui.run_button(label="Train Combined Best")
    train_combined
    return (train_combined,)


@app.cell
def _(mo, train_combined, train_model):
    mo.stop(not train_combined.value)
    train_model(epochs=750, batch=8, mosaic=0.0, name="ppe_750_combined", seed=48)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Experiment Analysis

    comparison of all training runs in configurations.
    """
    )
    return


@app.cell
def _(Path, mo):
    runs_dir = Path("runs/saved_2")
    if not runs_dir.exists():
        runs_dir.mkdir(parents=True, exist_ok=True)
    run_names = [
        d.name
        for d in runs_dir.iterdir()
        if d.is_dir() and d.name.startswith("ppe_")
    ]
    if not run_names:
        mo.md("**No training runs found yet. Run experiments above first.**")
    return run_names, runs_dir


@app.cell
def _(pl, yaml):
    def load_run_data(run_dir):
        results = pl.read_csv(run_dir / "results.csv")
        with open(run_dir / "args.yaml") as f:
            args = yaml.safe_load(f)
        return results, args
    return (load_run_data,)


@app.cell
def _(load_run_data, mo, run_names, runs_dir):
    mo.stop(len(run_names) == 0)
    all_runs = {}
    for _name in sorted(run_names):
        _run_path = runs_dir / _name
        _results, _args = load_run_data(_run_path)
        all_runs[_name] = {"results": _results, "args": _args}
    return (all_runs,)


@app.cell
def _(PrettyTable, TABLES_STYLE, all_runs, mo):
    params_table = PrettyTable()
    params_table.field_names = [
        "Run",
        "Epochs",
        "Batch",
        "LR",
        "Dropout",
        "Mosaic",
    ]

    for _name, _data in sorted(all_runs.items()):
        _args = _data["args"]
        params_table.add_row(
            [
                _name.replace("ppe_", ""),
                _args["epochs"],
                _args["batch"],
                _args["lr0"],
                _args["dropout"],
                _args["mosaic"],
            ]
        )

    params_table.set_style(TABLES_STYLE)
    mo.md(f"""
    ### Training Parameters

    {params_table.get_string()}
    """)
    return


@app.cell
def _(PrettyTable, TABLES_STYLE, all_runs, mo):
    results_table = PrettyTable()
    results_table.field_names = [
        "Run",
        "Precision",
        "Recall",
        "mAP50",
        "mAP50-95",
        "Val Loss",
    ]

    for _name, _data in sorted(all_runs.items()):
        _final = _data["results"].tail(1)
        results_table.add_row(
            [
                _name.replace("ppe_", ""),
                f"{_final['metrics/precision(B)'][0]:.4f}",
                f"{_final['metrics/recall(B)'][0]:.4f}",
                f"{_final['metrics/mAP50(B)'][0]:.4f}",
                f"{_final['metrics/mAP50-95(B)'][0]:.4f}",
                f"{_final['val/box_loss'][0]:.4f}",
            ]
        )

    results_table.set_style(TABLES_STYLE)
    mo.md(f"""
    ### Final Results

    {results_table.get_string()}
    """)
    return


@app.cell
def _(all_runs, plt, sns):
    sns.set_style("whitegrid")

    _fig, _axes = plt.subplots(2, 2, figsize=(15, 10))

    _metrics = [
        ("metrics/precision(B)", "Precision", _axes[0, 0]),
        ("metrics/recall(B)", "Recall", _axes[0, 1]),
        ("metrics/mAP50(B)", "mAP50", _axes[1, 0]),
        ("metrics/mAP50-95(B)", "mAP50-95", _axes[1, 1]),
    ]

    for _metric_name, _label, _ax in _metrics:
        for _name, _data in sorted(all_runs.items()):
            _results = _data["results"]
            _ax.plot(
                _results["epoch"],
                _results[_metric_name],
                label=_name.replace("ppe_", ""),
                linewidth=2,
            )

        _ax.set_xlabel("Epoch")
        _ax.set_ylabel(_label)
        _ax.set_title(f"{_label} Over Training")
        _ax.legend(loc="best", fontsize=8)
        _ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(all_runs, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(15, 5))

    for _name, _data in sorted(all_runs.items()):
        _results = _data["results"]
        _axes[0].plot(
            _results["epoch"],
            _results["train/box_loss"],
            label=_name.replace("ppe_", ""),
            linewidth=2,
        )
        _axes[1].plot(
            _results["epoch"],
            _results["val/box_loss"],
            label=_name.replace("ppe_", ""),
            linewidth=2,
        )

    _axes[0].set_xlabel("Epoch")
    _axes[0].set_ylabel("Loss")
    _axes[0].set_title("Training Box Loss")
    _axes[0].legend(loc="best", fontsize=8)
    _axes[0].grid(True, alpha=0.3)

    _axes[1].set_xlabel("Epoch")
    _axes[1].set_ylabel("Loss")
    _axes[1].set_title("Validation Box Loss")
    _axes[1].legend(loc="best", fontsize=8)
    _axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(all_runs, np, plt):
    _fig, _ax = plt.subplots(figsize=(10, 6))

    _run_labels = []
    _map50_values = []
    _map50_95_values = []

    for _name, _data in sorted(all_runs.items()):
        _final = _data["results"].tail(1)
        _run_labels.append(_name.replace("ppe_", ""))
        _map50_values.append(_final["metrics/mAP50(B)"][0])
        _map50_95_values.append(_final["metrics/mAP50-95(B)"][0])

    _x = np.arange(len(_run_labels))
    _width = 0.35

    _ax.bar(_x - _width / 2, _map50_values, _width, label="mAP50", alpha=0.8)
    _ax.bar(_x + _width / 2, _map50_95_values, _width, label="mAP50-95", alpha=0.8)

    _ax.set_ylabel("mAP Score")
    _ax.set_title("Final mAP Comparison Across Runs")
    _ax.set_xticks(_x)
    _ax.set_xticklabels(_run_labels, rotation=45, ha="right")
    _ax.legend()
    _ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Metric Explanations

    **Precision**: What percentage of predicted PPE violations are actually violations. High precision means fewer false alarms.

    **Recall**: What percentage of actual violations did we catch. High recall means we don't miss violations.

    **mAP50**: Mean Average Precision at 50% IoU threshold. Measures detection accuracy when bounding boxes overlap at least 50% with ground truth. Standard metric for object detection.

    **mAP50-95**: Average of mAP at IoU thresholds from 50% to 95%. Stricter metric that requires more accurate bounding boxes. Better indicator of overall model quality.

    **Box Loss**: How well predicted bounding boxes match ground truth locations. Lower is better.

    **Class Loss**: Cross-entropy loss for object classification. Measures how confidently the model predicts the correct class.

    **DFL Loss**: Distribution Focal Loss for box regression. Helps with more accurate bounding box localization.
    """
    )
    return


@app.cell
def _(all_runs, mo, pl):
    comparison_df = pl.DataFrame(
        {
            "Run": [name.replace("ppe_", "") for name in sorted(all_runs.keys())],
            "Precision": [
                all_runs[name]["results"]["metrics/precision(B)"][-1]
                for name in sorted(all_runs.keys())
            ],
            "Recall": [
                all_runs[name]["results"]["metrics/recall(B)"][-1]
                for name in sorted(all_runs.keys())
            ],
            "mAP50": [
                all_runs[name]["results"]["metrics/mAP50(B)"][-1]
                for name in sorted(all_runs.keys())
            ],
            "mAP50-95": [
                all_runs[name]["results"]["metrics/mAP50-95(B)"][-1]
                for name in sorted(all_runs.keys())
            ],
            "BoxLoss": [
                all_runs[name]["results"]["val/box_loss"][-1]
                for name in sorted(all_runs.keys())
            ],
            "ClsLoss": [
                all_runs[name]["results"]["val/cls_loss"][-1]
                for name in sorted(all_runs.keys())
            ],
        }
    )

    mo.md(f"""
    ### Detailed Comparison Table

    {comparison_df}
    """)
    return


@app.cell
def _(all_runs, plt):
    _fig, _axes = plt.subplots(2, 3, figsize=(18, 10))
    _axes = _axes.flatten()

    _metrics_to_plot = [
        ("train/box_loss", "Training Box Loss"),
        ("val/box_loss", "Validation Box Loss"),
        ("train/cls_loss", "Training Class Loss"),
        ("val/cls_loss", "Validation Class Loss"),
        ("train/dfl_loss", "Training DFL Loss"),
        ("val/dfl_loss", "Validation DFL Loss"),
    ]

    for _idx, (_metric, _title) in enumerate(_metrics_to_plot):
        for _name, _data in sorted(all_runs.items()):
            _results = _data["results"]
            _axes[_idx].plot(
                _results["epoch"],
                _results[_metric],
                label=_name.replace("ppe_", ""),
                linewidth=2,
                alpha=0.7,
            )

        _axes[_idx].set_xlabel("Epoch")
        _axes[_idx].set_ylabel("Loss")
        _axes[_idx].set_title(_title)
        _axes[_idx].legend(loc="best", fontsize=7)
        _axes[_idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Confusion Matrices Comparison

    Confusion matrices show which classes the model confuses most often.
    """
    )
    return


@app.cell
def _(mpimg, plt, run_names, runs_dir):
    _n_runs = len(run_names)
    _fig, _axes = plt.subplots(_n_runs, 1, figsize=(12, 8 * _n_runs))
    if _n_runs == 1:
        _axes = [_axes]

    for _idx, _name in enumerate(sorted(run_names)):
        _cm_path = runs_dir / _name / "confusion_matrix_normalized.png"
        if _cm_path.exists():
            _img = mpimg.imread(str(_cm_path))
            _axes[_idx].imshow(_img)
            _axes[_idx].set_title(
                f"{_name.replace('ppe_', '')} - Confusion Matrix", fontsize=14
            )
            _axes[_idx].axis("off")

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Precision-Recall Curves

    Shows the tradeoff between precision and recall at different confidence thresholds.
    """
    )
    return


@app.cell
def _(mpimg, plt, run_names, runs_dir):
    _n_runs = len(run_names)
    _fig, _axes = plt.subplots(_n_runs, 1, figsize=(12, 6 * _n_runs))
    if _n_runs == 1:
        _axes = [_axes]

    for _idx, _name in enumerate(sorted(run_names)):
        _pr_path = runs_dir / _name / "BoxPR_curve.png"
        if _pr_path.exists():
            _img = mpimg.imread(str(_pr_path))
            _axes[_idx].imshow(_img)
            _axes[_idx].set_title(
                f"{_name.replace('ppe_', '')} - PR Curve", fontsize=14
            )
            _axes[_idx].axis("off")

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### F1 Score Curves

    F1 is the harmonic mean of precision and recall. Shows optimal confidence threshold.
    """
    )
    return


@app.cell
def _(mpimg, plt, run_names, runs_dir):
    _n_runs = len(run_names)
    _fig, _axes = plt.subplots(_n_runs, 1, figsize=(12, 6 * _n_runs))
    if _n_runs == 1:
        _axes = [_axes]

    for _idx, _name in enumerate(sorted(run_names)):
        _f1_path = runs_dir / _name / "BoxF1_curve.png"
        if _f1_path.exists():
            _img = mpimg.imread(str(_f1_path))
            _axes[_idx].imshow(_img)
            _axes[_idx].set_title(
                f"{_name.replace('ppe_', '')} - F1 Curve", fontsize=14
            )
            _axes[_idx].axis("off")

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Training Metrics Over Time

    Complete training history for each run.
    """
    )
    return


@app.cell
def _(mpimg, plt, run_names, runs_dir):
    _n_runs = len(run_names)
    _cols = 2
    _rows = (_n_runs + 1) // 2
    _fig, _axes = plt.subplots(_rows, _cols, figsize=(20, 10 * _rows))
    _axes = _axes.flatten() if _n_runs > 1 else [_axes]

    for _idx, _name in enumerate(sorted(run_names)):
        _results_path = runs_dir / _name / "results.png"
        if _results_path.exists():
            _img = mpimg.imread(str(_results_path))
            _axes[_idx].imshow(_img)
            _axes[_idx].set_title(
                f"{_name.replace('ppe_', '')}", fontsize=14, weight="bold"
            )
            _axes[_idx].axis("off")

    for _idx in range(_n_runs, len(_axes)):
        _axes[_idx].axis("off")

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Convergence Analysis

    Checking if models converged or still improving at the end of training.
    """
    )
    return


@app.cell
def _(PrettyTable, TABLES_STYLE, all_runs, mo, np):
    convergence_table = PrettyTable()
    convergence_table.field_names = [
        "Run",
        "Last 10 Avg mAP",
        "Best mAP",
        "Epochs to Best",
        "Still Improving",
    ]

    for _name, _data in sorted(all_runs.items()):
        _results = _data["results"]
        _map_values = _results["metrics/mAP50-95(B)"].to_numpy()
        _last_10_avg = np.mean(_map_values[-10:])
        _best_map = np.max(_map_values)
        _best_epoch = np.argmax(_map_values) + 1
        _still_improving = "Yes" if _best_epoch > len(_map_values) - 10 else "No"

        convergence_table.add_row(
            [
                _name.replace("ppe_", ""),
                f"{_last_10_avg:.4f}",
                f"{_best_map:.4f}",
                _best_epoch,
                _still_improving,
            ]
        )

    convergence_table.set_style(TABLES_STYLE)
    mo.md(f"""
    {convergence_table.get_string()}
    """)
    return


if __name__ == "__main__":
    app.run()
