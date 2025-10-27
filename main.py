import marimo

__generated_with = "0.17.0"
app = marimo.App(
    width="medium",
    layout_file="layouts/main.slides.json",
    auto_download=["ipynb"],
)


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
    from ultralytics import YOLO
    return Path, YOLO, cv2, np, plt


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


app._unparsable_cell(
    r"""
    some images are bad. like white shirt is a west, or blue working vest is not recognized (image 4), or too many people (image 4)
    """,
    name="_"
)


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
    # TRAINING THE MODEL
    return


@app.cell
def _(YOLO):
    # Load YOLOv8 nano model
    # Pre-trained on COCO dataset with 80 object classes
    model = YOLO("yolov8n.pt")

    # Display actual model information
    model.info()
    return (model,)


@app.cell
def _(dataset_path, model):
    # Section: Test pre-trained model on construction images


    test_img_path = dataset_path / "train" / "images"
    test_images = list(test_img_path.glob("*.jpg"))[:3]

    # Run inference on 3 images
    for _img_file in test_images:
        _results = model.predict(
            source=str(_img_file),
            conf=0.25,
            save=False,
            verbose=True,  # Show what it actually detects
        )

        # Results object contains actual detections
        _result = _results[0]
        print(f"\n{_img_file.name}: {len(_result.boxes)} objects detected")
    return (test_images,)


@app.cell
def _(cv2, model, plt, test_images):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, img_file in enumerate(test_images):
        # Run prediction with save=False but get the plotted result
        _results = model.predict(
            source=str(img_file), conf=0.25, save=False, verbose=False
        )

        # Get the plotted image with boxes
        _result = _results[0]
        plotted_img = _result.plot()  # This draws boxes on the image

        # Convert BGR to RGB for matplotlib
        plotted_img_rgb = cv2.cvtColor(plotted_img, cv2.COLOR_BGR2RGB)

        axes[idx].imshow(plotted_img_rgb)
        axes[idx].set_title(f"{img_file.name}\n{len(_result.boxes)} detections")
        axes[idx].axis("off")

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""# detects people, but poorly, not all, and does not detect safety equipment""")
    return


@app.cell
def _(mo):
    mo.md(r"""## here i added .yaml file which is not on github""")
    return


@app.cell
def _(dataset_path, yaml_path):
    # Section: Create correct data.yaml

    # Get absolute paths
    train_path = (dataset_path / 'train' / 'images').resolve()
    val_path = (dataset_path / 'valid' / 'images').resolve()
    test_path = (dataset_path / 'test' / 'images').resolve()

    # Create correct yaml content
    yaml_content = f"""path: {dataset_path.resolve()}
    train: train/images
    val: valid/images
    test: test/images

    nc: 10
    names:
      0: Hardhat
      1: Mask
      2: NO-Hardhat
      3: NO-Mask
      4: NO-Safety Vest
      5: Person
      6: Safety Cone
      7: Safety Vest
      8: machinery
      9: vehicle
    """

    # Write the corrected yaml
    with open(yaml_path, 'w') as _f:
        _f.write(yaml_content)

    print("Updated data.yaml:")
    print(yaml_content)

    # Verify paths exist
    print("\nVerifying paths:")
    print(f"Train exists: {train_path.exists()}")
    print(f"Val exists: {val_path.exists()}")
    print(f"Test exists: {test_path.exists()}")
    return


@app.cell
def _(dataset_path):
    # Section: Verify data.yaml configuration
    import yaml

    yaml_path = dataset_path / "data.yaml"

    with open(yaml_path, "r") as _f:
        data_config = yaml.safe_load(_f)

    print(f"Classes: {data_config['nc']}")
    print(f"Names: {data_config['names']}")
    return (yaml_path,)


@app.cell
def _(mo):
    mo.md(r"""# training on gpu""")
    return


@app.cell
def _(model, yaml_path):
    # Section: Train YOLOv8 on PPE dataset

    # Train the model
    # epochs: number of complete passes through the training data
    # imgsz: input image size (must match dataset)
    # batch: number of images processed together (adjust based on your GPU/RAM)
    # device: 0 for GPU, 'cpu' for CPU
    results = model.train(
        data=str(yaml_path),
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,  # Change to "cpu" if needed
        project="runs/train",
        name="ppe_detection",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## ok lets check""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""" """)
    return


@app.cell
def _(Path, YOLO, plt):
    # Section: Analyze training results

    _results_path = Path('runs/train/ppe_detection2')

    # Load the best trained model
    trained_model = YOLO(_results_path / 'weights' / 'best.pt')

    # Display training results plot
    import matplotlib.image as mpimg

    _fig, _axes = plt.subplots(2, 2, figsize=(15, 12))

    # Show key result images
    _images_to_show = [
        ('results.png', 'Training Metrics'),
        ('confusion_matrix_normalized.png', 'Confusion Matrix'),
        ('val_batch0_labels.jpg', 'Validation: Ground Truth'),
        ('val_batch0_pred.jpg', 'Validation: Predictions')
    ]

    for _idx, (_img_name, _title) in enumerate(_images_to_show):
        _img_path = _results_path / _img_name
        if _img_path.exists():
            _img = mpimg.imread(str(_img_path))
            _axes[_idx // 2, _idx % 2].imshow(_img)
            _axes[_idx // 2, _idx % 2].set_title(_title)
            _axes[_idx // 2, _idx % 2].axis('off')

    plt.tight_layout()
    plt.show()
    return mpimg, trained_model


@app.cell
def _():
    ## make it bigger
    return


@app.cell
def _(Path, mpimg, plt):
    # Section: Analyze training results - Large view

    _results_path = Path('runs/train/ppe_detection2')

    # Load the best trained model
    # trained_model = YOLO(_results_path / 'weights' / 'best.pt')


    # Show each important plot in full size, one at a time

    # 1. Training Metrics
    _fig1 = plt.figure(figsize=(20, 12))
    _img1 = mpimg.imread(str(_results_path / 'results.png'))
    plt.imshow(_img1)
    plt.title('Training Metrics (Loss, Precision, Recall, mAP)', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # 2. Confusion Matrix
    _fig2 = plt.figure(figsize=(16, 16))
    _img2 = mpimg.imread(str(_results_path / 'confusion_matrix_normalized.png'))
    plt.imshow(_img2)
    plt.title('Confusion Matrix (Normalized)', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # 3. Validation Comparison
    _fig3, _axes3 = plt.subplots(1, 2, figsize=(24, 12))
    _img_labels = mpimg.imread(str(_results_path / 'val_batch0_labels.jpg'))
    _img_preds = mpimg.imread(str(_results_path / 'val_batch0_pred.jpg'))

    _axes3[0].imshow(_img_labels)
    _axes3[0].set_title('Validation: Ground Truth Labels', fontsize=14)
    _axes3[0].axis('off')

    _axes3[1].imshow(_img_preds)
    _axes3[1].set_title('Validation: Model Predictions', fontsize=14)
    _axes3[1].axis('off')

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(Path, np):
    # Section: Interactive training results visualization with Bokeh

    from bokeh.plotting import figure, show, output_notebook
    from bokeh.layouts import column
    from PIL import Image
    # import numpy as np

    output_notebook()

    _results_path = Path('runs/train/ppe_detection2')

    def show_image_bokeh(img_path, title, width=1200, height=800):
        """Display image with Bokeh - properly"""
        _img = np.array(Image.open(img_path))
    
        # Convert to RGBA if needed
        if _img.ndim == 2:  # grayscale
            _img_rgba = np.stack([_img, _img, _img, np.full(_img.shape, 255, dtype=np.uint8)], axis=2)
        elif _img.shape[2] == 3:  # RGB
            _img_rgba = np.dstack([_img, np.full(_img.shape[:2], 255, dtype=np.uint8)])
        else:  # already RGBA
            _img_rgba = _img
    
        # Flip vertically for Bokeh
        _img_rgba = np.flipud(_img_rgba)
    
        # Convert to uint32 view
        _img_uint32 = _img_rgba.view(dtype=np.uint32).reshape(_img_rgba.shape[:2])
    
        _h, _w = _img_uint32.shape
    
        _p = figure(
            width=width,
            height=height,
            title=title,
            x_range=(0, _w),
            y_range=(0, _h),
            tools="pan,wheel_zoom,box_zoom,reset,save"
        )
    
        _p.image_rgba(image=[_img_uint32], x=0, y=0, dw=_w, dh=_h)
        _p.axis.visible = False
    
        return _p

    # Create plots
    _p1 = show_image_bokeh(_results_path / 'results.png', 'Training Metrics', 1400, 900)
    _p2 = show_image_bokeh(_results_path / 'confusion_matrix_normalized.png', 'Confusion Matrix', 1000, 1000)
    _p3 = show_image_bokeh(_results_path / 'val_batch0_labels.jpg', 'Ground Truth', 1200, 800)
    _p4 = show_image_bokeh(_results_path / 'val_batch0_pred.jpg', 'Predictions', 1200, 800)

    # Show all
    show(column(_p1, _p2, _p3, _p4))
    return


@app.cell
def _(cv2, dataset_path, model, plt, trained_model):
    def _():
        # Section: Compare pre-trained vs fine-tuned model performance

        test_img_path = dataset_path / 'test' / 'images'
        test_images_comparison = list(test_img_path.glob('*.jpg'))[:4]

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))

        for idx, img_file in enumerate(test_images_comparison):
            # Pre-trained COCO model
            results_pretrained = model.predict(str(img_file), conf=0.25, verbose=False)
            img_pretrained = results_pretrained[0].plot()
            img_pretrained_rgb = cv2.cvtColor(img_pretrained, cv2.COLOR_BGR2RGB)
        
            # Fine-tuned PPE model
            results_finetuned = trained_model.predict(str(img_file), conf=0.25, verbose=False)
            img_finetuned = results_finetuned[0].plot()
            img_finetuned_rgb = cv2.cvtColor(img_finetuned, cv2.COLOR_BGR2RGB)
        
            # Display
            axes[0, idx].imshow(img_pretrained_rgb)
            axes[0, idx].set_title(f'COCO: {len(results_pretrained[0].boxes)} detections', fontsize=10)
            axes[0, idx].axis('off')
        
            axes[1, idx].imshow(img_finetuned_rgb)
            axes[1, idx].set_title(f'PPE: {len(results_finetuned[0].boxes)} detections', fontsize=10)
            axes[1, idx].axis('off')

        fig.text(0.02, 0.75, 'Pre-trained\n(COCO)', fontsize=14, weight='bold', va='center', rotation=90)
        fig.text(0.02, 0.25, 'Fine-tuned\n(PPE)', fontsize=14, weight='bold', va='center', rotation=90)

        plt.tight_layout()
        return plt.show()


    _()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
