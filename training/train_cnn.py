"""
training/train_cnn.py
──────────────────────
Fine-tunes ResNet-50 (ImageNet pretrained) for hydroponic plant
growth stage classification.

Classes (4):  seedling | vegetative | flowering | harvest
Input:        224 × 224 RGB plant images
Architecture: ResNet-50 backbone → GlobalAvgPool → Dense(256) → Dense(4)

Dataset layout expected:
  data/plant_images/
      seedling/    *.jpg
      vegetative/  *.jpg
      flowering/   *.jpg
      harvest/     *.jpg

If the dataset folder is absent, a synthetic dummy dataset (coloured blobs)
is generated automatically so the script always runs end-to-end.

Output saved to:  saved_models/resnet50_growth.pt

Run:
  python training/train_cnn.py [--epochs 20] [--batch 32] [--lr 1e-4]
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("saved_models",          exist_ok=True)
os.makedirs("training/outputs",      exist_ok=True)
os.makedirs("data/plant_images",     exist_ok=True)


# ─── Synthetic dataset fallback ───────────────────────────────────────────────

def _make_synthetic_dataset(root: str, n_per_class: int = 200):
    """
    Generate simple synthetic images (colour-coded by growth stage)
    so the training pipeline can run without real plant photos.
    Each image is a 224×224 numpy array with Gaussian blobs.
    """
    from PIL import Image, ImageDraw
    import random

    COLOURS = {
        "seedling":   (210, 240, 180),
        "vegetative": (80,  180, 60),
        "flowering":  (220, 80,  160),
        "harvest":    (240, 200, 50),
    }

    for stage, colour in COLOURS.items():
        stage_dir = os.path.join(root, stage)
        os.makedirs(stage_dir, exist_ok=True)
        existing = len(os.listdir(stage_dir))
        for i in range(existing, existing + n_per_class):
            img = Image.new("RGB", (224, 224), color=(30, 30, 30))
            draw = ImageDraw.Draw(img)
            for _ in range(random.randint(4, 10)):
                x = random.randint(20, 200)
                y = random.randint(20, 200)
                r = random.randint(8, 30)
                c = tuple(max(0, min(255, v + random.randint(-30, 30))) for v in colour)
                draw.ellipse([x - r, y - r, x + r, y + r], fill=c)
            img.save(os.path.join(stage_dir, f"{stage}_{i:04d}.jpg"))
    print(f"  Synthetic images generated in {root}")


# ─── Model Definition ─────────────────────────────────────────────────────────

def build_model(n_classes: int = 4, freeze_backbone: bool = True):
    import torch
    import torch.nn as nn
    from torchvision import models

    backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    if freeze_backbone:
        # Freeze all layers except the last 3 convolutional blocks
        layers_to_freeze = list(backbone.children())[:-3]
        for layer in layers_to_freeze:
            for p in layer.parameters():
                p.requires_grad = False

    in_features = backbone.fc.in_features
    backbone.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, n_classes),
    )
    return backbone


# ─── Training ─────────────────────────────────────────────────────────────────

def train(
    epochs:           int   = 20,
    batch:            int   = 32,
    lr:               float = 1e-4,
    data_root:        str   = "data/plant_images",
    n_synthetic:      int   = 200,
):
    print("=" * 60)
    print("  SmartHydro — ResNet-50 Growth Stage Trainer")
    print("=" * 60)

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, random_split
    from torchvision import datasets, transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    # ── 1. Dataset ─────────────────────────────────────────────────
    print(f"\n[1/5] Preparing dataset from {data_root}…")
    has_images = any(
        os.listdir(os.path.join(data_root, c))
        for c in ["seedling", "vegetative", "flowering", "harvest"]
        if os.path.isdir(os.path.join(data_root, c))
    ) if os.path.isdir(data_root) else False

    if not has_images:
        print("  No images found — generating synthetic dataset…")
        _make_synthetic_dataset(data_root, n_per_class=n_synthetic)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    full_dataset = datasets.ImageFolder(data_root)
    n_val   = max(int(len(full_dataset) * 0.15), 1)
    n_test  = max(int(len(full_dataset) * 0.15), 1)
    n_train = len(full_dataset) - n_val - n_test

    train_set, val_set, test_set = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )
    train_set.dataset.transform = train_tf
    val_set.dataset.transform   = val_tf
    test_set.dataset.transform  = val_tf

    train_loader = DataLoader(train_set, batch_size=batch, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_set,  batch_size=batch, shuffle=False, num_workers=2)

    classes = full_dataset.classes
    print(f"  Classes: {classes}")
    print(f"  Train: {n_train} | Val: {n_val} | Test: {n_test}")

    # ── 2. Model ───────────────────────────────────────────────────
    print("\n[2/5] Building ResNet-50 model…")
    model = build_model(n_classes=len(classes)).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── 3. Training loop ───────────────────────────────────────────
    print(f"\n[3/5] Training ({epochs} epochs)…")
    best_val_acc = 0.0
    history      = {"train_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        scheduler.step()
        train_loss = running_loss / n_train

        # Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds   = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
        val_acc = correct / n_val

        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model, "saved_models/resnet50_growth.pt")

        if epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:3d}/{epochs} | Loss {train_loss:.4f} | Val Acc {val_acc:.3f}")

    # ── 4. Test set evaluation ─────────────────────────────────────
    print("\n[4/5] Evaluating on test set…")
    model = torch.load("saved_models/resnet50_growth.pt", map_location=device)
    model.eval()
    correct   = 0
    class_tp  = {c: 0 for c in classes}
    class_tot = {c: 0 for c in classes}

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            for pred, label in zip(preds.cpu(), labels.cpu()):
                cn = classes[label]
                class_tot[cn] += 1
                if pred == label:
                    class_tp[cn] += 1

    test_acc = correct / n_test
    print(f"\n  Overall test accuracy: {test_acc:.3f}")
    for c in classes:
        acc = class_tp[c] / max(class_tot[c], 1)
        print(f"    {c:<12} {acc:.3f}")

    # ── 5. Plots ───────────────────────────────────────────────────
    print("\n[5/5] Saving plots…")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"]); ax1.set_title("Train Loss"); ax1.set_xlabel("Epoch")
    ax2.plot(history["val_acc"]);    ax2.set_title("Val Accuracy"); ax2.set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig("training/outputs/cnn_training.png", dpi=150)
    plt.close()

    print("\n  Plot saved   → training/outputs/cnn_training.png")
    print("  Model saved  → saved_models/resnet50_growth.pt")
    print(f"\n  Best val accuracy: {best_val_acc:.3f}")
    print("\nDone ✅")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet-50 growth stage classifier")
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--batch",       type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--data",        type=str,   default="data/plant_images")
    parser.add_argument("--n_synthetic", type=int,   default=200,
                        help="Synthetic images per class if no real data found")
    args = parser.parse_args()
    train(epochs=args.epochs, batch=args.batch, lr=args.lr,
          data_root=args.data, n_synthetic=args.n_synthetic)
