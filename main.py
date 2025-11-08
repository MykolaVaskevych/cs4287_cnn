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
    return


@app.cell
def _():
    import random
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import cv2
    from pathlib import Path
    import numpy as np
    from ultralytics import YOLO
    import yaml
    from collections import Counter, defaultdict
    from bokeh.plotting import figure, show, output_notebook
    from bokeh.layouts import column
    from PIL import Image
    import torch
    import math
    import json
    import shutil
    return (
        Image,
        Path,
        YOLO,
        column,
        cv2,
        defaultdict,
        figure,
        json,
        mpimg,
        np,
        output_notebook,
        plt,
        random,
        show,
        shutil,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # CS4287-CNN: Construction Safety Equipment Detection

    **Authors**: MYKOLA VASKEVYCH (22372199), OLIVER FITZGERALD (22365958)

    **Status**: Code executes to completion: YES

    ## Overview
    This notebook fine-tunes a YOLOv8 nano model to detect Personal Protective Equipment (PPE)
    violations on construction sites. The model identifies safety equipment (hardhats, masks,
    safety vests) and flags violations when workers lack proper protection.

    ## Dataset
    - **Classes**: 10 (Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest, Person, Safety Cone, Safety Vest, machinery, vehicle)
    - **Format**: YOLO format with normalized bounding boxes
    - **Splits**: Train/Validation/Test

    ## Quick Start
    1. Ensure dataset is in `data/archive/css-data/` directory
    2. Run all cells sequentially (training will not start automatically)
    3. Review dataset statistics and quality
    4. Click "Train Model" button when ready to train
    5. Scroll down to see training results and model comparison
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # CHECKS & SETTINGS
    ## NOTE: CHECK CONSTANTS BELOW TO ENSURE CORRECTNESS BEFORE RUNNING THE NOTEBOOK.

    **⚡ Quick Navigation**: [Jump to Train Button](#training-configuration)
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


@app.cell
def _(Path, np, random, torch):
    # Reproducibility
    RANDOM_SEED = 42
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    # Auto-detect best available device
    if torch.cuda.is_available():
        DEVICE = 0
        _device_name = torch.cuda.get_device_name(0)
        _vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU detected: {_device_name} ({_vram_gb:.1f}GB VRAM)")
        print(f"Recommended BATCH_SIZE: {16 if _vram_gb >= 8 else 8}")
    else:
        DEVICE = "cpu"
        print("⚠ No GPU detected - training will be significantly slower")
        print("Recommended: Reduce EPOCHS to 10 and BATCH_SIZE to 4 for CPU")

    # Dataset paths
    DATASET_ROOT = Path.cwd() / "data" / "archive" / "css-data"
    TRAINING_IMAGES_PATH = (DATASET_ROOT / "train" / "images").resolve()
    SAMPLE = (Path.cwd() / "sample/").resolve() # Tempporary Line Delete
    TRAINING_LABELS_PATH = (DATASET_ROOT / "train" / "labels").resolve()
    VALIDATION_IMAGES_PATH = (DATASET_ROOT / "valid" / "images").resolve()
    VALIDATION_LABELS_PATH = (DATASET_ROOT / "valid" / "labels").resolve()
    TEST_IMAGES_PATH = (DATASET_ROOT / "test" / "images").resolve()
    TEST_LABELS_PATH = (DATASET_ROOT / "test" / "labels").resolve()
    YAML_CONFIG_PATH = DATASET_ROOT / "data.yaml"

    # Model paths
    PRETRAINED_MODEL_PATH = "yolov8n.pt"
    TRAINING_OUTPUT_DIR = "runs/train"
    TRAINING_RUN_NAME = "ppe_detection4"
    TRAINED_MODEL_PATH = (
        Path(TRAINING_OUTPUT_DIR) / TRAINING_RUN_NAME / "weights" / "best.pt"
    )

    # Training parameters
    EPOCHS = 100  # Reduce to 10 for quick testing or CPU training
    IMAGE_SIZE = 640  # YOLO standard input size
    BATCH_SIZE = 16  # Reduce to 4-8 for low VRAM or CPU

    # Detection parameters
    CONFIDENCE_THRESHOLD = 0.25  # Minimum confidence for detections (0.0-1.0)

    # Visualization parameters
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
    return (
        BATCH_SIZE,
        BBOX_COLORS,
        CLASS_NAMES,
        CONFIDENCE_THRESHOLD,
        DATASET_ROOT,
        DEVICE,
        EPOCHS,
        IMAGE_SIZE,
        NUM_BASELINE_TEST_SAMPLES,
        NUM_COMPARISON_IMAGES,
        PRETRAINED_MODEL_PATH,
        SAMPLE,
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
def _(
    BATCH_SIZE,
    CONFIDENCE_THRESHOLD,
    DATASET_ROOT,
    DEVICE,
    EPOCHS,
    IMAGE_SIZE,
    TEST_IMAGES_PATH,
    TRAINING_IMAGES_PATH,
    VALIDATION_IMAGES_PATH,
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

    print("✓ All dataset paths validated successfully\n")

    # Display configuration summary
    mo.md(
        f"""
        ## Current Configuration

        | Parameter | Value | Description |
        |-----------|-------|-------------|
        | Device | `{DEVICE}` | Training device (0=GPU, 'cpu'=CPU) |
        | Epochs | `{EPOCHS}` | Training iterations through dataset |
        | Image Size | `{IMAGE_SIZE}px` | Input resolution |
        | Batch Size | `{BATCH_SIZE}` | Images per training step |
        | Confidence | `{CONFIDENCE_THRESHOLD}` | Min score for detections |

        **Note**: Adjust BATCH_SIZE in constants cell if you get OOM (Out of Memory) errors.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Dataset
    ## Dataset Configuration

    Generates the YOLO-compatible data.yaml configuration file
    """
    )
    return


@app.cell
def _(CLASS_NAMES, DATASET_ROOT, YAML_CONFIG_PATH, regenerate_yaml):
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

    # Only update YAML if it doesn't exist or content differs
    if regenerate_yaml.value:
        with open(YAML_CONFIG_PATH, "w") as _f:
            _f.write(_yaml_content)
        print("✓ Generated data.yaml:")
        print(_yaml_content)
    else:
        print("✓ Skipped YAML generation (using existing file)")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Dataset Inspection

    Examine the distribution of images and labels across train/validation/test splits.
    """
    )
    return


@app.cell
def _(DATASET_ROOT, Path, json, random, shutil):
    print("=" * 50)
    print("DATASET STRUCTURE")
    print("=" * 50)

    # Re-Structure Dataset to 80:10:10 Train:Test:Validation Split
    # File extensionsed that are moved
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}

    class FileGrouping:
        def __init__(self, filePath, labelPath):
            self.filePath = filePath
            self.labelPath = labelPath

    class Configuration:
        def __init__(self,configuration_path):

            fileType = configuration_path[len(configuration_path) - 5:]
            if fileType != ".json":
                raise ValueError(f"Given filetype \"{fileType}\" But must be of type \"*.json\"")

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
                reprString += " - " + target["target path"] + " (" + target["distribution"] + " of total images)\n"

            return reprString

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
        global image_extensions

        # Verify directories exist
        if not all(directory.exists() for directory in configuration.source):
            print(f"Error: One or more of the specified source directories don't exist! {configuration.source}")
            return
        if not all(Path(target["target path"]).exists() for target in configuration.target):
            print(f"Error: One or more of the specified target directories don't exist! {configuration.source}")
            return


        # Collect all images from all directories
        all_images = []
        currentDistribution = {}
        for directory in configuration.source:
            count = 0
            for file_path in directory.rglob('*'):

                relative_path = file_path.relative_to(directory)
                imagePath = directory / relative_path  # actual image path
                labelPath = Path(str(imagePath)
                                 .replace('/images/', '/labels/', 1)
                                 .rsplit('.', 1)[0] + '.txt')

                if imagePath.is_file() and labelPath.is_file():

                    # Create FileGrouping object
                    file_grouping = FileGrouping(imagePath, labelPath)
                    all_images.append(file_grouping)
                    count += 1
            currentDistribution[directory] = count

        total_images = len(all_images)
        if total_images == 0:
            raise ValueError(f"No images in the given source paths: {configuration.source}")

        print(f"Found {total_images} total images")
        print("Current Distribution:")
        for directory, fileCount in currentDistribution.items():
            print(f"{" " * 4}{directory} contains {fileCount} files constituting ({fileCount / total_images}) of the full distribution (1.0)")

        # Shuffle images randomly
        print("Shuffiling images ...")
        random.shuffle(all_images)

        # Calculate split indices
        print("\nMoving images...")
        for target in configuration.target:
            images = all_images[:round(float(target["distribution"]) * total_images)] 
            all_images = all_images[round(float(target["distribution"]) * total_images):] 
            move_images(images, Path(target["target path"]))


        print("✓ Redistribution complete!")

        print(f"\nNew distribution:")
        for target in configuration.target:
            print(f"{" " * 4}{target["target path"]} contains {float(target["distribution"]) * total_images} files constituting ({float(target["distribution"]) * total_images / total_images}) of the full distribution (1.0)")

    def move_images(image_list, target_dir):
        """
        Moves a list of FileGrouping objects (images and labels) to given target directories
        Args:
            image_list (list): List of FileGrouping objects to be moved
            target_dir: The base target directory (images go to target_dir/images, labels to target_dir/labels)
        """
        # Create target subdirectories for images and labels
        target_images_dir = target_dir / 'images'
        target_labels_dir = target_dir / 'labels'

        for file_grouping in image_list:
            # Move image file
            img_path = file_grouping.filePath
            print(f"image_path: {img_path}\ntarget_img_path: {target_images_dir}\n")
            target_img_path = target_images_dir / img_path.name
            if img_path.exists() and img_path.parent != target_images_dir:
                shutil.move(str(img_path), str(target_images_dir))

            # Move label file
            label_path = file_grouping.labelPath
            print(f"label_path: {label_path}\ntarget_img_path: {target_labels_dir}\n")
            target_label_path = target_labels_dir / label_path.name
            if label_path.exists() and label_path.parent != target_labels_dir:
                shutil.move(str(label_path), str(target_labels_dir))

    try:
        configuration = Configuration("config.json")
        redistribute_images(configuration)
    except ValueError as e:
        print(f"Invalid Configuration Data: {e}")



    # Examine Datastructure 
    for _split in ["train", "valid", "test"]:
        _img_path = DATASET_ROOT / _split / "images"
        _label_path = DATASET_ROOT / _split / "labels"

        _num_images = len(list(_img_path.glob("*.jpg")))
        _num_labels = len(list(_label_path.glob("*.txt")))

        print(f"{_split.upper():10s}: {_num_images} images, {_num_labels} labels")
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
def _(TRAINING_LABELS_PATH):
    print("=" * 50)
    print("SAMPLE LABEL FILE")
    print("=" * 50)

    _label_files = list(TRAINING_LABELS_PATH.glob("*.txt"))
    _first_label = _label_files[0]

    print(f"\nSample Label file: {_first_label.name}")
    print("\nContents (first 10 lines):")
    print("\nFormat: class_id x_center y_center width height")
    with open(_first_label, "r") as _f:
        _lines = _f.readlines()[:10]
        for _i, _line in enumerate(_lines, 1):
            print(f"  {_i}. {_line.strip()}")

    print(f"\nTotal objects in this image: {len(_lines)}")
    print("(Note: All values normalized between 0 and 1)")
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
def _(DATASET_ROOT):
    print("=" * 50)
    print("CLASS DISTRIBUTION")
    print("=" * 50)

    _class_counts = {}
    _total_objects = 0

    for _split in ["train", "valid", "test"]:
        _label_path = DATASET_ROOT / _split / "labels"

        for _label_file in _label_path.glob("*.txt"):
            with open(_label_file, "r") as _f:
                for _line in _f:
                    _parts = _line.strip().split()
                    if _parts:
                        _class_id = int(_parts[0])
                        _class_counts[_class_id] = _class_counts.get(_class_id, 0) + 1
                        _total_objects += 1

    print(f"\nTotal objects across all classes: {_total_objects}")
    print(f"Number of unique classes: {len(_class_counts)}")
    print("\nClass distribution:")
    for _class_id in sorted(_class_counts.keys()):
        _count = _class_counts[_class_id]
        _percentage = (_count / _total_objects) * 100
        print(f"  Class {_class_id}: {_count:5d} objects ({_percentage:5.2f}%)")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Sample Images Visualization

    Display training images with ground truth bounding boxes overlaid.
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
def _(BBOX_COLORS, CLASS_NAMES, SAMPLE, draw_boxes_on_image, plt):
    print("=" * 50)
    print("VISUALIZING SAMPLE IMAGES")
    print("=" * 50)

    NUM_SAMPLE_IMAGES = 6 # Minimum == 3

    _image_files = list(SAMPLE.glob("*.jpg"))[:NUM_SAMPLE_IMAGES]
    _fig, _axes = plt.subplots(NUM_SAMPLE_IMAGES // 3, 3, figsize=(15, 10))
    _axes = _axes.flatten()

    for _idx, _img_file in enumerate(_image_files):
        _label_file = SAMPLE / (_img_file.stem + ".txt")

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

    # --- Added code (displays each image individually) ---
    """"
    for _idx, _img_file in enumerate(_image_files):
        _label_file = SAMPLE / (_img_file.stem + ".txt")
        if _label_file.exists():
            _img_with_boxes = draw_boxes_on_image(
                _img_file, _label_file, CLASS_NAMES, BBOX_COLORS
            )
            plt.figure(figsize=(5, 5))
            plt.imshow(_img_with_boxes)
            plt.title(f"Individual View: {_img_file.name}")
            plt.axis("off")
            plt.show()
    """

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
    # Load YOLOv8 nano model pre-trained on COCO dataset (80 classes)
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
    # CRITICAL: Use TEST set for unbiased baseline evaluation
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
    <span id="training-configuration"></span>

    ## Training Configuration

    Training progress will be displayed below. Key metrics to monitor:
    - **mAP50**: Mean Average Precision at IoU=0.5 (higher is better, target >0.7)
    - **Loss**: Should decrease steadily over epochs
    - **Precision/Recall**: Balance between false positives and false negatives

    **Estimated time**: 30-60 minutes on GPU, 4-8 hours on CPU (50 epochs)

    Click the button below when ready to start training.
    """
    )
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
    ## Interactive Training Results

    Zoomable Bokeh visualizations of training results with pan and zoom capabilities.
    """
    )
    return


@app.cell
def _(
    Image,
    TRAINED_MODEL_PATH,
    column,
    figure,
    mo,
    np,
    output_notebook,
    show,
):
    mo.stop(
        not TRAINED_MODEL_PATH.exists(),
        mo.md("**Train the model first to see interactive results.**"),
    )

    output_notebook()

    _results_dir = TRAINED_MODEL_PATH.parent.parent


    def _show_image_bokeh(img_path, title, width=1200, height=800):
        """Display image with Bokeh for interactive exploration"""
        _img = np.array(Image.open(img_path))

        # Convert to RGBA
        if _img.ndim == 2:
            _img_rgba = np.stack(
                [_img, _img, _img, np.full(_img.shape, 255, dtype=np.uint8)],
                axis=2,
            )
        elif _img.shape[2] == 3:
            _img_rgba = np.dstack(
                [_img, np.full(_img.shape[:2], 255, dtype=np.uint8)]
            )
        else:
            _img_rgba = _img

        _img_rgba = np.flipud(_img_rgba)
        _img_uint32 = _img_rgba.view(dtype=np.uint32).reshape(_img_rgba.shape[:2])

        _h, _w = _img_uint32.shape

        _p = figure(
            width=width,
            height=height,
            title=title,
            x_range=(0, _w),
            y_range=(0, _h),
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )

        _p.image_rgba(image=[_img_uint32], x=0, y=0, dw=_w, dh=_h)
        _p.axis.visible = False

        return _p


    _p1 = _show_image_bokeh(
        _results_dir / "results.png", "Training Metrics", 1400, 900
    )
    _p2 = _show_image_bokeh(
        _results_dir / "confusion_matrix_normalized.png",
        "Confusion Matrix",
        1000,
        1000,
    )
    _p3 = _show_image_bokeh(
        _results_dir / "val_batch0_labels.jpg", "Ground Truth", 1200, 800
    )
    _p4 = _show_image_bokeh(
        _results_dir / "val_batch0_pred.jpg", "Predictions", 1200, 800
    )

    show(column(_p1, _p2, _p3, _p4))
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


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Using the Trained Model

    To use the trained model on new images:

    ```python
    from ultralytics import YOLO

    # Load trained model
    model = YOLO("runs/train/ppe_detection/weights/best.pt")

    # Run inference
    results = model.predict(
        source="path/to/image.jpg",
        conf=0.25,
        save=True,
        save_txt=True  # Save labels in YOLO format
    )

    # Get detections
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            print(f"Detected: {CLASS_NAMES[cls]} (confidence: {conf:.2f})")
    ```

    The model will save annotated images to `runs/detect/predict/`.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Troubleshooting

    **Q: Training fails with CUDA out of memory**
    - Reduce BATCH_SIZE in constants cell (try 8, then 4)
    - Reduce IMAGE_SIZE to 416
    - Close other GPU-intensive applications

    **Q: Training is very slow**
    - Check DEVICE is set to GPU (should see "GPU detected" message)
    - If on CPU, reduce EPOCHS to 10 for faster iteration
    - Ensure CUDA drivers are properly installed

    **Q: Poor detection performance (low mAP)**
    - Check class distribution - severe imbalance may need data augmentation
    - Increase EPOCHS (try 100)
    - Try larger YOLO models (yolov8s.pt, yolov8m.pt)
    - Verify dataset labels are correct

    **Q: Model doesn't exist error**
    - Click "Train Model" button and wait for training to complete
    - Check that TRAINING_RUN_NAME matches the actual folder in `runs/train/`
    - Verify TRAINED_MODEL_PATH points to correct location

    **Q: Dataset not found error**
    - Ensure dataset is extracted to `data/archive/css-data/`
    - Check directory structure matches expected layout
    - Verify all splits (train/valid/test) exist

    **Q: Import errors or missing packages**
    - Run: `uv sync` to install all dependencies
    - Check that you're using Python 3.13+
    - Try: `uv add torch` if GPU detection fails
    """
    )
    return


if __name__ == "__main__":
    app.run()
