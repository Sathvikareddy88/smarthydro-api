"""
training/train_yolo.py
───────────────────────
Fine-tunes YOLOv8-S on hydroponic pest and disease images.

Detection classes (12):
  aphid | whitefly | powdery_mildew | leaf_chlorosis | tip_burn |
  botrytis | root_rot | nutrient_deficiency | spider_mite |
  fusarium | downy_mildew | healthy

Dataset layout (YOLO format):
  data/pest_detection/
      images/train/*.jpg
      images/val/*.jpg
      labels/train/*.txt     (YOLO format: class cx cy w h)
      labels/val/*.txt
      dataset.yaml

If no dataset is found, a minimal synthetic YOLO dataset is generated
(coloured rectangles with bounding-box annotations).

Output saved to:  saved_models/yolov8_pest.pt

Run:
  python training/train_yolo.py [--epochs 100] [--imgsz 640] [--batch 16]
"""

import os
import argparse
import shutil
import yaml

os.makedirs("saved_models",              exist_ok=True)
os.makedirs("training/outputs",          exist_ok=True)
os.makedirs("data/pest_detection/images/train", exist_ok=True)
os.makedirs("data/pest_detection/images/val",   exist_ok=True)
os.makedirs("data/pest_detection/labels/train", exist_ok=True)
os.makedirs("data/pest_detection/labels/val",   exist_ok=True)

CLASSES = [
    "aphid", "whitefly", "powdery_mildew", "leaf_chlorosis", "tip_burn",
    "botrytis", "root_rot", "nutrient_deficiency", "spider_mite",
    "fusarium", "downy_mildew", "healthy",
]


# ─── Synthetic dataset generator ─────────────────────────────────────────────

def _make_synthetic_yolo_dataset(
    root: str,
    n_train: int = 300,
    n_val:   int = 60,
):
    """
    Generate synthetic YOLO-format images with random bounding boxes.
    Each image has 1–3 annotated objects.
    """
    import random
    import numpy as np
    from PIL import Image, ImageDraw

    CLASS_COLOURS = {
        "aphid":              (180, 50,  50),
        "whitefly":           (230, 230, 230),
        "powdery_mildew":     (240, 240, 200),
        "leaf_chlorosis":     (220, 220, 80),
        "tip_burn":           (200, 130, 60),
        "botrytis":           (160, 80,  160),
        "root_rot":           (100, 60,  40),
        "nutrient_deficiency":(50,  200, 200),
        "spider_mite":        (220, 100, 50),
        "fusarium":           (130, 60,  190),
        "downy_mildew":       (170, 200, 170),
        "healthy":            (60,  190, 60),
    }

    def make_image_and_labels(split, idx, img_size=640):
        img  = Image.new("RGB", (img_size, img_size), (40, 80, 40))
        draw = ImageDraw.Draw(img)
        lines = []

        n_objs = random.randint(1, 3)
        for _ in range(n_objs):
            cls_name = random.choice(CLASSES)
            cls_idx  = CLASSES.index(cls_name)
            colour   = CLASS_COLOURS[cls_name]

            # Random bounding box
            w  = random.randint(30, 120)
            h  = random.randint(30, 120)
            x1 = random.randint(0, img_size - w)
            y1 = random.randint(0, img_size - h)
            x2, y2 = x1 + w, y1 + h

            # Draw object (tinted blob)
            for _ in range(random.randint(3, 8)):
                bx = random.randint(x1, x2)
                by = random.randint(y1, y2)
                br = random.randint(4, 15)
                c  = tuple(max(0, min(255, v + random.randint(-30, 30))) for v in colour)
                draw.ellipse([bx - br, by - br, bx + br, by + br], fill=c)

            # YOLO label: class cx cy w h (normalised)
            cx = (x1 + x2) / 2 / img_size
            cy = (y1 + y2) / 2 / img_size
            nw = w / img_size
            nh = h / img_size
            lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        fname = f"synthetic_{idx:05d}"
        img.save(os.path.join(root, "images", split, f"{fname}.jpg"))
        with open(os.path.join(root, "labels", split, f"{fname}.txt"), "w") as f:
            f.write("\n".join(lines))

    print(f"  Generating {n_train} train + {n_val} val synthetic images…")
    for i in range(n_train):
        make_image_and_labels("train", i)
    for i in range(n_val):
        make_image_and_labels("val", i)
    print("  Synthetic dataset generated.")


def _write_dataset_yaml(root: str):
    cfg = {
        "path":  os.path.abspath(root),
        "train": "images/train",
        "val":   "images/val",
        "nc":    len(CLASSES),
        "names": CLASSES,
    }
    path = os.path.join(root, "dataset.yaml")
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return path


# ─── Training ─────────────────────────────────────────────────────────────────

def train(
    epochs:     int = 100,
    imgsz:      int = 640,
    batch:      int = 16,
    data_root:  str = "data/pest_detection",
    pretrained: str = "yolov8s.pt",   # YOLOv8-S pretrained weights
):
    print("=" * 60)
    print("  SmartHydro — YOLOv8 Pest Detection Trainer")
    print("=" * 60)

    from ultralytics import YOLO

    # ── 1. Prepare dataset ─────────────────────────────────────────
    print(f"\n[1/4] Preparing dataset in {data_root}…")
    train_imgs = os.listdir(os.path.join(data_root, "images", "train"))
    if len(train_imgs) < 10:
        _make_synthetic_yolo_dataset(data_root)

    yaml_path = os.path.join(data_root, "dataset.yaml")
    if not os.path.exists(yaml_path):
        yaml_path = _write_dataset_yaml(data_root)
    print(f"  Dataset YAML: {yaml_path}")

    # ── 2. Load base model ─────────────────────────────────────────
    print(f"\n[2/4] Loading YOLOv8-S base model ({pretrained})…")
    model = YOLO(pretrained)

    # ── 3. Fine-tune ───────────────────────────────────────────────
    print(f"\n[3/4] Fine-tuning ({epochs} epochs, imgsz={imgsz}, batch={batch})…")
    results = model.train(
        data          = yaml_path,
        epochs        = epochs,
        imgsz         = imgsz,
        batch         = batch,
        lr0           = 1e-3,
        lrf           = 0.01,
        momentum      = 0.937,
        weight_decay  = 5e-4,
        warmup_epochs = 3,
        mosaic        = 1.0,
        mixup         = 0.15,
        copy_paste    = 0.1,
        degrees       = 10.0,
        fliplr        = 0.5,
        hsv_h         = 0.015,
        hsv_s         = 0.7,
        hsv_v         = 0.4,
        patience      = 15,
        save          = True,
        project       = "training/outputs/yolo_runs",
        name          = "pest_detection",
        exist_ok      = True,
        verbose       = True,
    )

    # ── 4. Export best weights ─────────────────────────────────────
    print("\n[4/4] Exporting best weights…")
    best_weights = "training/outputs/yolo_runs/pest_detection/weights/best.pt"
    if os.path.exists(best_weights):
        shutil.copy(best_weights, "saved_models/yolov8_pest.pt")
        print("  Weights copied → saved_models/yolov8_pest.pt")

    # Print key metrics
    try:
        metrics = results.results_dict
        print(f"\n  mAP@0.5:     {metrics.get('metrics/mAP50(B)',   'N/A')}")
        print(f"  mAP@0.5:0.95: {metrics.get('metrics/mAP50-95(B)','N/A')}")
        print(f"  Precision:   {metrics.get('metrics/precision(B)','N/A')}")
        print(f"  Recall:      {metrics.get('metrics/recall(B)',    'N/A')}")
    except Exception:
        pass

    print("\nDone ✅")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 for pest detection")
    parser.add_argument("--epochs",     type=int, default=100)
    parser.add_argument("--imgsz",      type=int, default=640)
    parser.add_argument("--batch",      type=int, default=16)
    parser.add_argument("--data",       type=str, default="data/pest_detection")
    parser.add_argument("--pretrained", type=str, default="yolov8s.pt")
    args = parser.parse_args()
    train(epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
          data_root=args.data, pretrained=args.pretrained)
