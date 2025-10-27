import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    print("hello world")
    return


@app.cell
def _():
    # # CS4287-CNN
    # Names: MYKOLA VASKEVYCH(22372199), Teammate Name (ID2)
    # Code executes to completion: YES

    import os
    import matplotlib.pyplot as plt
    import cv2
    from pathlib import Path
    import numpy as np
    return Path, cv2, plt


@app.cell
def _(Path):
    print(Path.cwd())
    return


@app.cell
def _(Path):
    # Set your dataset path
    """
    [nick@archlinux css-data]$ pwd
    /home/nick/Documents/projects/uni_stuff/cnn/cs4287_cnn/data/archive/css-data
    [nick@archlinux css-data]$ ls
    README.dataset.txt  README.roboflow.txt  test  train  valid
    """

    dataset_path = Path.cwd() / "data" / "archive" / "css-data"

    # Section 1: Understanding Dataset Structure
    print("=" * 50)
    print("DATASET STRUCTURE")
    print("=" * 50)

    # Count files in each split
    for _split in ["train", "valid", "test"]:
        img_path = Path(dataset_path) / _split / "images"
        _label_path = Path(dataset_path) / _split / "labels"

        num_images = len(list(img_path.glob("*.jpg")))
        num_labels = len(list(_label_path.glob("*.txt")))

        print(f"{_split.upper():10s}: {num_images} images, {num_labels} labels")
    return (dataset_path,)


@app.cell
def _(Path, dataset_path):
    # Section 2: Examine Label Format
    print("\n" + "=" * 50)
    print("SAMPLE LABEL FILE")
    print("=" * 50)

    # Get first training label file
    label_files = list((Path(dataset_path) / "train" / "labels").glob("*.txt"))
    first_label = label_files[0]

    print(f"\nLabel file: {first_label.name}")
    print("\nContents (first 10 lines):")
    with open(first_label, "r") as _f:
        lines = _f.readlines()[:10]  # Read first 10 lines
        for i, _line in enumerate(lines, 1):
            print(f"  {i}. {_line.strip()}")

    print(f"\nTotal objects in this image: {len(lines)}")
    print("\nFormat: class_id x_center y_center width height")
    print("(All values normalized between 0 and 1)")
    return


@app.cell
def _(Path, dataset_path):
    # Section 3: Discover Class Names
    print("\n" + "=" * 50)
    print("CLASS DISCOVERY")
    print("=" * 50)

    from collections import Counter

    # Collect all class IDs from all label files
    all_class_ids = []

    for _split in ["train", "valid", "test"]:
        _label_path = Path(dataset_path) / _split / "labels"

        for label_file in _label_path.glob("*.txt"):
            with open(label_file, "r") as _f:
                for _line in _f:
                    parts = _line.strip().split()
                    if parts:  # If line is not empty
                        class_id = int(parts[0])
                        all_class_ids.append(class_id)

    # Count occurrences of each class
    class_counts = Counter(all_class_ids)

    print(f"\nTotal objects across all images: {len(all_class_ids)}")
    print(f"Number of unique classes: {len(class_counts)}")
    print("\nClass distribution:")
    for class_id in sorted(class_counts.keys()):
        count = class_counts[class_id]
        percentage = (count / len(all_class_ids)) * 100
        print(f"  Class {class_id}: {count:5d} objects ({percentage:5.2f}%)")
    return


@app.cell
def _(cv2, dataset_path, plt):
    # TODO: REPLACE with yaml file?? or leave like that?

    # Class names
    _class_names = {
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

    # Colors for each class (BGR format for OpenCV)
    _colors = {
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


    def draw_boxes_on_image(img_path, label_path, class_names, colors):
        """Draw bounding boxes on image"""
        # Read image
        _img = cv2.imread(str(img_path))
        # print(str(img_path))
        _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
        _h, _w = _img.shape[:2]
        # print(_h,_w)

        # print(str(label_path))

        # Read labels
        with open(label_path, "r") as _f:
            for _line in _f:
                _parts = _line.strip().split()
                if not _parts:
                    continue

                # print(_parts)
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

                # Add labelds
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


    # Section 4: Visualize Images with Bounding Boxes
    print("=" * 50)
    print("VISUALIZING SAMPLE IMAGES")
    print("=" * 50)

    _train_img_path = dataset_path / "train" / "images"
    _train_label_path = dataset_path / "train" / "labels"

    # Get first 6 images
    _image_files = list(_train_img_path.glob("*.jpg"))[
        :6
    ]  # <---------------- some images are bad, check [4:5] (1,2..)
    # print(_image_files)                                                    |
    _fig, _axes = plt.subplots(2, 3, figsize=(15, 10))  # <-------------------|
    _axes = _axes.flatten()

    for _idx, _img_file in enumerate(_image_files):
        _label_file = _train_label_path / (_img_file.stem + ".txt")

        if _label_file.exists():
            _img_with_boxes = draw_boxes_on_image(
                _img_file, _label_file, _class_names, _colors
            )
            _axes[_idx].imshow(_img_with_boxes)
            _axes[_idx].set_title(f"Image: {_img_file.name}", fontsize=10)
            _axes[_idx].axis("off")

    plt.tight_layout()
    plt.show()

    print("\nLegend:")
    print("  Green boxes: Hardhat, Safety Vest (PPE worn correctly)")
    print("  Red boxes: NO-Hardhat, NO-Mask, NO-Safety Vest (violations!)")
    print("  Magenta: Person")
    print("  Orange: Safety Cone")
    print("  Gray: Machinery")
    print("  Blue: Vehicle")
    return


@app.cell
def _(mo):
    mo.md(
        r"""some images are bad. like white shirt is a west, or blue working vest is not recognized (image 4), or too many people (image 4)"""
    )
    return


@app.cell
def _(dataset_path):
    print("=" * 50)
    print("DATA QUALITY OBSERVATIONS")
    print("=" * 50)

    # Analyze class co-occurrences to spot issues
    from collections import defaultdict

    _cooccurrence = defaultdict(int)
    _dataset_path = dataset_path

    # Check train set
    _train_label_path = _dataset_path / "train" / "labels"

    for _label_file in _train_label_path.glob("*.txt"):
        _classes_in_image = set()
        with open(_label_file, "r") as _f:
            for _line in _f:
                _parts = _line.strip().split()
                if _parts:
                    _classes_in_image.add(int(_parts[0]))

        # Check for suspicious combinations
        if 5 in _classes_in_image:  # Person present
            if 7 in _classes_in_image and 4 in _classes_in_image:
                _cooccurrence["Person with BOTH vest AND no-vest"] += 1
            if 0 in _classes_in_image and 2 in _classes_in_image:
                _cooccurrence["Person with BOTH hardhat AND no-hardhat"] += 1

    print("\nPotential labeling inconsistencies found:")
    for _issue, _count in _cooccurrence.items():
        print(f"  {_issue}: {_count} images")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
