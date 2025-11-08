# Oliver Fitzgerald (22365958) & Mykola Vaskevych (22372199)
# The code executes to the end without an error
# Link to third party implmentation used: TODO-Inlude-Link

import marimo

__generated_with = "0.17.2"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell
def _():
    import marimo as mo
    import random
    import torch
    import math
    import json
    import shutil
    import cv2
    import yaml
    import hashlib
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import numpy as np
    import seaborn as sns
    from pathlib import Path
    from ultralytics import YOLO
    from collections import Counter, defaultdict
    from bokeh.plotting import figure, show, output_notebook
    from bokeh.layouts import column
    from PIL import Image
    from prettytable import PrettyTable
    from prettytable import TableStyle
    import polars as pl
    return (
        Counter,
        Path,
        PrettyTable,
        TableStyle,
        YOLO,
        cv2,
        hashlib,
        json,
        mo,
        mpimg,
        np,
        pl,
        plt,
        random,
        shutil,
        sns,
        torch,
        yaml,
    )


@app.cell(hide_code=True)
def _():
    def ascii_table_to_latex(ascii_table, caption="", label="", precision=3):
        """
        tmp for latex stuff, dont forget to change table style to default or ascii
        Note: this method is to aid in the construction of latex tables
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

    **Authors**: MYKOLA VASKEVYCH (22372199), OLIVER FITZGERALD (22365958)<br>
    **Status**: Code executes to completion: <span style="color: green">YES</span><br>
    **Third Party Implmentation**: [Ultralytcs YOLO8](https://www.google.com)


    ## Overview
    This notebook fine-tunes a YOLOv8 nano model to detect Personal Protective Equipment (PPE)
    violations on construction sites. The model identifies safety equipment (hardhats, masks,
    safety vests, etc.) and flags violations when workers lack proper protection.

    ## Dataset
    - **Classes**: 10 (Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest, Person, Safety Cone, Safety Vest, machinery, vehicle)
    - **Format**: YOLO format with normalized bounding boxes
    - **Splits**: Train/Validation/Test

    ## Quick Start
    1. Ensure dataset is in `data/archive/css-data/` directory<br>
    Data can be dowloaded from the following [kaggle page](https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow) in the follwing format:
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

    3. Run all cells sequentially (training will not start automatically) ????
    4. Review dataset statistics and quality to get an overview of the data and model
    5. Click the [Train Model](#training-configuration) when ready to commence train
    6. Scroll down to see training results and model comparison
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Checks & Settings
    **NOTE**: Training parameters below should be validated before running the notebook for optimal performance.<br>
    The follwing section contains pre-configured training parameters for everything from training device to dataset locations and configuration of model parameters along with descriptors where appropriate.
    """
    )
    return


@app.cell
def _(Path, PrettyTable, TableStyle, np, random, torch):
    """ "
    Reproducibility
    This section sets the random seeds used in each library throughout the notebook to allow for randomness while ensuring reproducibility
    """

    RANDOM_SEED = 42
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    """"
    Traing Device
    This section uses PyTorch to detect the best available training device. If a CUDA driver for an NVIDIA GPU is found, that device will be used; otherwise, training will run on the CPU.  
    You can override this autodetection by setting 'DEVICE_OVERRIDE' to your preferred device — e.g., "cpu" or '0' for CUDA.
    """
    DEVICE_OVERRIDE = None

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    # Auto-detect best available device i.e CPU or CUDA Drivers
    if DEVICE_OVERRIDE is not None:
        DEVICE = DEVICE_OVERRIDE
        print(f"Auto-Detect best device overrided using device: {DEVICE}")
    elif torch.cuda.is_available():
        DEVICE = 0
        _device_name = torch.cuda.get_device_name(0)
        _vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU detected: {_device_name} ({_vram_gb:.1f}GB VRAM)")
        print(f"Recommended BATCH_SIZE: {16 if _vram_gb >= 8 else 8}")
    else:
        DEVICE = "cpu"
        print("No GPU detected - training will be significantly slower")
        print("Recommended: Reduce EPOCHS to 10 and BATCH_SIZE to 4 for CPU")

    """"
    Dataset Paths
    This section specifies the directories within the project where the dataset and related configuration files are located.  
    Note: These paths should generally not be modified unless you know what you are doing.
    """
    DATASET_ROOT = Path.cwd() / "data" / "archive" / "css-data"
    TRAINING_IMAGES_PATH = (DATASET_ROOT / "train" / "images").resolve()
    SAMPLE_IMAGES = (Path.cwd() / "sample/").resolve()
    TRAINING_LABELS_PATH = (DATASET_ROOT / "train" / "labels").resolve()
    VALIDATION_IMAGES_PATH = (DATASET_ROOT / "valid" / "images").resolve()
    VALIDATION_LABELS_PATH = (DATASET_ROOT / "valid" / "labels").resolve()
    TEST_IMAGES_PATH = (DATASET_ROOT / "test" / "images").resolve()
    TEST_LABELS_PATH = (DATASET_ROOT / "test" / "labels").resolve()
    YAML_CONFIG_PATH = DATASET_ROOT / "data.yaml"

    """
    Model Paths
    This section defines the locations of the pre-trained model and the output paths for the model generated during training.
    """
    PRETRAINED_MODEL_PATH = "yolov8n.pt"
    TRAINING_OUTPUT_DIR = "runs/train"
    TRAINING_RUN_NAME = "ppe_detection4"
    TRAINED_MODEL_PATH = (
        Path(TRAINING_OUTPUT_DIR) / TRAINING_RUN_NAME / "weights" / "best.pt"
    )

    """"
    Training Parameters
    Defines the main hyperparameters for model training.
    Note: May need to be altered depending on hardware specifications for more optimal training times
    """
    EPOCHS = 100  # Reduce to 10 for quick testing or CPU training
    IMAGE_SIZE = 640  # Standard image size for YOLOv8 and this dataset
    BATCH_SIZE = 16  # Reduce to 4-8 for low VRAM or CPU

    # Detection parameters
    CONFIDENCE_THRESHOLD = 0.25  # Minimum confidence for detections (0.0-1.0)

    """
    Dataset Visualization parameters
    This section defines settings for how dataset samples and annotations are displayed for inspection and analysis.
    """
    TABLES_STYLE = TableStyle.MARKDOWN
    NUM_COMPARISON_IMAGES = 4
    NUM_BASELINE_TEST_SAMPLES = 3

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

    # Bounding box colors (BGR format for OpenCV)
    BBOX_COLORS = {
        0: (0, 255, 0),  # Hardhat - Green
        1: (255, 255, 0),  # Mask - Cyan
        2: (0, 0, 255),  # NO-Hardhat - Red
        3: (0, 0, 255),  # NO-Mask - Red
        4: (0, 0, 255),  # NO-Safety Vest - Red
        5: (255, 0, 255),  # Person - Magenta
        6: (0, 165, 255),  # Safety Cone - Orange
        7: (0, 255, 0),  # Safety Vest - Green
        8: (128, 128, 128),  # machinery - Gray
        9: (255, 0, 0),  # vehicle - Blue
    }


    """
    Display configuration summary
    """
    _table = PrettyTable()
    _table.field_names = ["Parameter", "Value", "Description"]
    _table.add_rows(
        [
            ["Device", f"{DEVICE}", "Training device (0=GPU, 'cpu'=CPU)"],
            ["Epochs", f"{EPOCHS}", "Training iterations through dataset"],
            ["Image Size", f"{IMAGE_SIZE}px", "Input resolution"],
            ["Batch Size", f"{BATCH_SIZE}", "Images per training step"],
            ["Confidence", f"{CONFIDENCE_THRESHOLD}", "Min score for detections"],
        ]
    )
    _table.set_style(TABLES_STYLE)
    print("Current Configuration")
    print(_table)
    print(
        "\nNote: Adjust BATCH_SIZE in constants cell if you get OOM (Out of Memory) errors."
    )
    return (
        BBOX_COLORS,
        CLASS_NAMES,
        CONFIDENCE_THRESHOLD,
        DATASET_ROOT,
        NUM_BASELINE_TEST_SAMPLES,
        PRETRAINED_MODEL_PATH,
        SAMPLE_IMAGES,
        TABLES_STYLE,
        TEST_IMAGES_PATH,
        TRAINING_IMAGES_PATH,
        TRAINING_LABELS_PATH,
        VALIDATION_IMAGES_PATH,
        YAML_CONFIG_PATH,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    # Dataset
    ## Dataset Configuration

    Generates the YOLO-compatible data.yaml configuration file with the parameters defined in the previous seciton.
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
    mo,
):
    # Validate dataset is present and structure is correct
    _missing_paths = []
    _paths_to_check = {
        "Dataset root": DATASET_ROOT,
        "Training images": TRAINING_IMAGES_PATH,
        "Validation images": VALIDATION_IMAGES_PATH,
        "Test images": TEST_IMAGES_PATH,
    }

    for _name, _path in _paths_to_check.items():
        if not _path.exists():
            _missing_paths.append(f"- {_name}: `{_path}`")

    if _missing_paths:
        _error_msg = "**ERROR: Missing required paths:**\n\n" + "\n".join(
            _missing_paths
        )
        _error_msg += (
            "\n\n**Please ensure dataset is extracted to the correct location.**"
        )
        mo.stop(True, mo.md(_error_msg))

    print("All dataset paths validated successfully\n")

    # Generate YAML using CLASS_NAMES constant
    _names_yaml = "\n  ".join([f"{k}: {v}" for k, v in CLASS_NAMES.items()])

    _yaml_content = f"""path: {DATASET_ROOT.resolve()}
    train: train/images
    val: valid/images
    test: test/images

    nc: {len(CLASS_NAMES)}
    names:
      {_names_yaml}
    """

    with open(YAML_CONFIG_PATH, "w") as _f:
        _f.write(_yaml_content)
    print("Generated data.yaml:")
    print(_yaml_content)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Dataset Distribution

    Examine the distribution of images and labels across train/validation/test splits and makes any corrections required if the split is not of the desired ratio i.e (80:10:10) across (train:test:valid)
    """
    )
    return


@app.cell
def _(DATASET_ROOT, Path, hashlib, json, random, shutil):
    print("=" * 50)
    print("DATASET STRUCTURE")
    print("Re-Structures Dataset to 80:10:10 Train:Test:Validation Split")
    print("=" * 50)


    class FileGrouping:
        def __init__(self, filePath, labelPath):
            self.filePath = filePath
            self.labelPath = labelPath


    class Configuration:
        """
        Configuration
        An object containg relevant information from a passed json configuration file which defines the desired sturcture of the dataset
        """

        def __init__(self, configuration_path):
            fileType = configuration_path[len(configuration_path) - 5 :]
            if fileType != ".json":
                raise ValueError(
                    f'Given filetype "{fileType}" But must be of type "*.json"'
                )

            with open(configuration_path, "r") as file:
                fileContent = json.load(file)

                self.source = [Path(path) for path in fileContent["source paths"]]
                self.target = fileContent["targets"]

        def __repr__(self):
            reprString = "Source Folders\n"
            for source in self.source:
                reprString += " - " + source + "\n"
            reprString += "\n"

            reprString += "Target Folders\n"
            for target in self.target:
                reprString += (
                    " - "
                    + target["target path"]
                    + " ("
                    + target["distribution"]
                    + " of total images)\n"
                )

            return reprString


    def get_file_hash(file_path):
        """Calculate MD5 hash of a file's contents."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()


    def redistribute_images(configuration):
        """
        Redistributes files from a set of source directories to a set of target directories
        according to a specified distribution.

        Args:
            configuration (dict): Dictionary specifying the redistribution operation:
                - 'source paths': List of source directories containing files.
                - 'targets': List of dictionaries, each with:
                    - 'target path': Path to the target directory.
                    - 'distribution': Fraction (string or float) of files to move to this target.
        """

        # Verify directories exist
        if not all(directory.exists() for directory in configuration.source):
            print(
                f"Error: One or more of the specified source directories don't exist! {configuration.source}"
            )
            return
        if not all(
            Path(target["target path"]).exists() for target in configuration.target
        ):
            print(
                f"Error: One or more of the specified target directories don't exist! {configuration.source}"
            )
            return

        # Collect all images from all directories
        all_images = []
        currentDistribution = {}
        seen_image_hashes = {}

        for directory in configuration.source:
            directory = directory / "images"
            count = 0
            for file_path in directory.rglob("*"):
                relative_path = file_path.relative_to(directory)
                imagePath = directory / relative_path
                labelPath = Path(
                    str(imagePath)
                    .replace("/images/", "/labels/", 1)
                    .rsplit(".", 1)[0]
                    + ".txt"
                )

                image_hash = get_file_hash(imagePath)
                if image_hash in seen_image_hashes:
                    print(f"Duplicate image found: {imagePath}")
                    print(f"Original: {seen_image_hashes[image_hash]}")
                    print()
                    continue
                if imagePath.is_file() and labelPath.is_file():
                    file_grouping = FileGrouping(imagePath, labelPath)
                    all_images.append(file_grouping)
                    seen_image_hashes[image_hash] = imagePath
                    count += 1

            currentDistribution[directory] = count

        total_images = len(all_images)
        if total_images == 0:
            raise ValueError(
                f"No images in the given source paths: {configuration.source}"
            )

        print(f"Found {total_images} total images")
        print("Current Distribution:")
        for directory, fileCount in currentDistribution.items():
            print(
                f"{' ' * 4}{directory} contains {fileCount} files constituting ({fileCount / total_images}) of the full distribution (1.0)"
            )

        # Shuffle images randomly
        print(f"\nShuffiling images ...")
        random.shuffle(all_images)

        # Re-distribute shuffeled images accross specified split
        print("Moving images...\n")
        for target in configuration.target:
            images = all_images[
                : round(float(target["distribution"]) * total_images)
            ]
            all_images = all_images[
                round(float(target["distribution"]) * total_images) :
            ]
            move_images(images, Path(target["target path"]))

        print("Redistribution complete!")

        print(f"\nNew distribution:")
        for target in configuration.target:
            print(
                f"{' ' * 4}{target['target path']} contains {float(target['distribution']) * total_images} files constituting ({float(target['distribution']) * total_images / total_images}) of the full distribution (1.0)"
            )


    def move_images(image_list, target_dir):
        """
        Moves a list of FileGrouping objects (images and labels) to given target directories
        Args:
            image_list (list): List of FileGrouping objects to be moved
            target_dir: The base target directory the FileGrouping object gets sendt to i.e (the image goes to target_dir/images and the labels go to target_dir/labels)
        """
        target_images_dir = target_dir / "images"
        target_labels_dir = target_dir / "labels"

        for file_grouping in image_list:
            # Move image file
            img_path = file_grouping.filePath
            if img_path.exists() and img_path.parent != target_images_dir:
                shutil.move(str(img_path), str(target_images_dir))

            # Move label file
            label_path = file_grouping.labelPath
            if label_path.exists() and label_path.parent != target_labels_dir:
                shutil.move(str(label_path), str(target_labels_dir))


    try:
        configuration = Configuration("config.json")
        redistribute_images(configuration)
    except ValueError as e:
        print(f"Invalid Configuration Data: {e}")


    print()
    # Finally Examine Resulting Datastructure
    for _split in ["train", "valid", "test"]:
        _img_path = DATASET_ROOT / _split / "images"
        _label_path = DATASET_ROOT / _split / "labels"
    return


@app.cell
def _(YOLO):
    def train_model(
        epochs,
        batch,
        lr0=0.01,
        dropout=0.0,
        mosaic=0.0,
        name="experiment",
        project="runs/saved/",
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
            optimizer="ADAM",
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
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Label Analysis

    The following cell gives an example of how labels are formated for each image
    YOLO format uses: 

    `class_id x_center y_center width height` (normalized 0-1).

    Example:
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
    Note: All values normalized between 0 and 1
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
    _total_issues = sum(len(_v) for _v in validation_issues.values())
    clean_btn = mo.ui.run_button(label="Clean Dataset")
    clean_btn
    return (clean_btn,)


@app.cell
def _(DATASET_ROOT, Path, clean_btn, mo, validation_issues):
    mo.stop(not clean_btn.value)

    _has_issues = any(len(_v) > 0 for _v in validation_issues.values())

    if not _has_issues:
        mo.md("*Dataset is already clean - no issues to fix*")

    mo.stop(not _has_issues)

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
    mo.md(
        r"""
    ## Sample Images Visualization

    Display sample training images with ground truth bounding boxes overlaid.
    """
    )
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
    BBOX_COLORS,
    CLASS_NAMES,
    SAMPLE_IMAGES,
    TRAINING_IMAGES_PATH,
    TRAINING_LABELS_PATH,
    draw_boxes_on_image,
    plt,
):
    print("=" * 50)
    print("VISUALIZING SAMPLE IMAGES")
    print("=" * 50)

    NUM_SAMPLE_IMAGES = 6  # Minimum == 3

    """
    Displays a set number of images with all classes defined from the training set
    """
    _image_files = list(TRAINING_IMAGES_PATH.glob("*.jpg"))[:NUM_SAMPLE_IMAGES]
    _fig, _axes = plt.subplots(NUM_SAMPLE_IMAGES // 3, 3, figsize=(15, 10))
    _axes = _axes.flatten()

    for _idx, _img_file in enumerate(_image_files):
        _label_file = TRAINING_LABELS_PATH / (_img_file.stem + ".txt")

        if _label_file.exists():
            _img_with_boxes = draw_boxes_on_image(
                _img_file, _label_file, CLASS_NAMES, BBOX_COLORS
            )
            if _idx >= len(_axes):
                break

            _axes[_idx].imshow(_img_with_boxes)
            _axes[_idx].set_title(f"Image: {_img_file.name}", fontsize=10)
            _axes[_idx].axis("off")
    plt.tight_layout()
    plt.show()


    print("=" * 50)
    print("VISUALIZING SAMPLE Classes")
    print("=" * 50)


    """
    Displays an image for each class where that class is isocolated to display an example of how said class appears within an image
    """
    _all_image_files = list(SAMPLE_IMAGES.glob("*.jpg"))
    _fig, _axes = plt.subplots(2, 5, figsize=(20, 8))
    _axes = _axes.flatten()

    for _idx, _img_file in enumerate(_all_image_files):
        _label_file = SAMPLE_IMAGES / (_img_file.stem + ".txt")

        if _label_file.exists():
            _img_with_boxes = draw_boxes_on_image(
                _img_file, _label_file, CLASS_NAMES, BBOX_COLORS
            )
            if _idx >= len(_axes):
                break

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
    mo.md(r"""### Baseline (100 epochs, batch=16, lr=0.000714) # true rate 714 not default whatever""")
    return


@app.cell
def _(mo):
    train_baseline = mo.ui.run_button(
        label="Train Baseline",
    )
    train_baseline
    return (train_baseline,)


@app.cell
def _(mo, train_baseline, train_model):
    mo.stop(not train_baseline.value) # true rate 714 not default whaterever
    train_model(epochs=100, batch=16, name="ppe_100", seed=42)
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
    mo.md(r"""### Learning Rate 0.001""")
    return


@app.cell
def _(mo):
    train_lr001 = mo.ui.run_button(label="Train LR=0.001")  # 00012 folder, true rate 001
    train_lr001
    return (train_lr001,)


@app.cell
def _(mo, train_lr001, train_model):
    mo.stop(not train_lr001.value)
    train_model(epochs=100, batch=16, lr0=0.001, name="ppe_100_lr_0.001", seed=45)
    return


@app.cell
def _(mo):
    mo.md(r"""<!-- ### Learning Rate 0.02 -->""")
    return


@app.cell
def _():
    # train_lr002 = mo.ui.run_button(label="Train LR=0.02")
    # train_lr002
    return


@app.cell
def _():
    # mo.stop(not train_lr002.value)
    # train_model(epochs=100, batch=16, lr0=0.02, name="ppe_100_lr_0.02", seed=46)
    return


@app.cell
def _(mo):
    mo.md(r"""### Dropout 0.2 (adma 0.01)""")
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
    runs_dir = Path("runs/saved")
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
    print(all_runs)
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
def _(PrettyTable, TABLES_STYLE, all_runs, mo, pl):
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


    _table = PrettyTable()
    _table.field_names = comparison_df.columns
    _table.add_rows(comparison_df.rows())
    _table.set_style(TABLES_STYLE)

    mo.md(f"""
    ### Detailed Comparison Table
    {_table.get_string()}
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
