"""
training/train_autoencoder.py
──────────────────────────────
Trains an undercomplete Autoencoder to learn the compressed representation
of normal hydroponic system behaviour.

At inference, high reconstruction error → operational anomaly.

Architecture:
  Encoder:  Dense(64, ReLU) → Dense(32, ReLU) → Dense(16, ReLU)  [bottleneck]
  Decoder:  Dense(32, ReLU) → Dense(64, ReLU) → Dense(10, Linear)

Trained ONLY on normal (non-anomalous) data.
Threshold = 95th percentile of training reconstruction errors.

Output saved to:
  saved_models/autoencoder.h5
  saved_models/autoencoder_threshold.npy

Run:
  python training/train_autoencoder.py [--epochs 80] [--batch 256]
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("saved_models",     exist_ok=True)
os.makedirs("training/outputs", exist_ok=True)


def build_autoencoder(input_dim: int = 10):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    inp = keras.Input(shape=(input_dim,), name="input")

    # Encoder
    x = layers.Dense(64,  activation="relu", name="enc_1")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(32,  activation="relu", name="enc_2")(x)
    x = layers.BatchNormalization()(x)
    z = layers.Dense(16,  activation="relu", name="bottleneck")(x)   # latent

    # Decoder
    x = layers.Dense(32,  activation="relu", name="dec_1")(z)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64,  activation="relu", name="dec_2")(x)
    out = layers.Dense(input_dim, activation="linear", name="output")(x)

    autoencoder = keras.Model(inp, out, name="Autoencoder")
    autoencoder.compile(
        optimizer = keras.optimizers.Adam(learning_rate=1e-3),
        loss      = "mse",
    )
    return autoencoder


def train(epochs: int = 80, batch: int = 256):
    print("=" * 60)
    print("  SmartHydro — Autoencoder Anomaly Detector Trainer")
    print("=" * 60)

    import tensorflow as tf
    from tensorflow import keras
    from sklearn.preprocessing import StandardScaler
    import joblib

    # ── 1. Generate data ──────────────────────────────────────────
    print("\n[1/6] Generating synthetic operational data…")
    from training.data_generators.hydro_data import (
        generate_multi_crop, build_autoencoder_dataset
    )
    df = generate_multi_crop(n_days=90)
    print(f"  Total rows: {len(df):,} | Anomalies: {df['is_anomaly'].sum():,}")

    # ── 2. Build dataset ──────────────────────────────────────────
    print("[2/6] Building normal / anomaly datasets…")
    X_normal, X_anomaly = build_autoencoder_dataset(df)
    print(f"  Normal: {len(X_normal):,} | Anomaly: {len(X_anomaly):,}")

    # ── 3. Scale (fit on normal only) ─────────────────────────────
    print("[3/6] Scaling features (StandardScaler on normal data)…")
    scaler   = StandardScaler()
    X_normal = scaler.fit_transform(X_normal)
    X_anomaly = scaler.transform(X_anomaly)
    joblib.dump(scaler, "saved_models/autoencoder_scaler.pkl")

    # Train/val split (normal data only)
    n_val   = int(len(X_normal) * 0.15)
    X_train = X_normal[:-n_val].astype("float32")
    X_val   = X_normal[-n_val:].astype("float32")
    print(f"  Train: {len(X_train):,} | Val: {len(X_val):,}")

    # ── 4. Train ──────────────────────────────────────────────────
    print(f"\n[4/6] Training Autoencoder ({epochs} epochs, batch={batch})…")
    input_dim = X_train.shape[1]
    model     = build_autoencoder(input_dim)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6),
        keras.callbacks.ModelCheckpoint("saved_models/autoencoder.h5", save_best_only=True),
    ]

    history = model.fit(
        X_train, X_train,
        validation_data = (X_val, X_val),
        epochs          = epochs,
        batch_size      = batch,
        callbacks       = callbacks,
        verbose         = 1,
    )

    # ── 5. Compute threshold ──────────────────────────────────────
    print("\n[5/6] Computing anomaly threshold…")
    model  = keras.models.load_model("saved_models/autoencoder.h5")
    recon  = model.predict(X_train, verbose=0)
    errors = np.mean((X_train - recon) ** 2, axis=1)

    threshold = float(np.percentile(errors, 95))
    np.save("saved_models/autoencoder_threshold.npy", threshold)
    print(f"  Threshold (95th pct): {threshold:.6f}")

    # Evaluate anomaly detection on held-out anomaly samples
    recon_anom  = model.predict(X_anomaly[:1000], verbose=0)
    errors_anom = np.mean((X_anomaly[:1000] - recon_anom) ** 2, axis=1)

    tp = int(np.sum(errors_anom  > threshold))
    fp_check = int(np.sum(errors  > threshold))
    precision = tp / max(tp + fp_check, 1)
    recall    = tp / max(len(errors_anom), 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)

    print(f"\n  Anomaly Detection (on held-out anomaly set):")
    print(f"    True Positives: {tp} / {len(errors_anom)}")
    print(f"    F1-Score:       {f1:.3f}")

    # ── 6. Plots ───────────────────────────────────────────────────
    print("\n[6/6] Saving plots…")

    # Loss curve
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"],     label="Train MSE")
    plt.plot(history.history["val_loss"], label="Val MSE")
    plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.title("Autoencoder Loss"); plt.legend()

    # Error distribution
    plt.subplot(1, 2, 2)
    plt.hist(errors,      bins=60, alpha=0.6, label="Normal",  color="steelblue")
    plt.hist(errors_anom, bins=60, alpha=0.6, label="Anomaly", color="tomato")
    plt.axvline(threshold, color="black", linestyle="--", label=f"Threshold={threshold:.4f}")
    plt.xlabel("Reconstruction Error"); plt.ylabel("Count")
    plt.title("Error Distribution"); plt.legend()

    plt.tight_layout()
    plt.savefig("training/outputs/autoencoder_training.png", dpi=150)
    plt.close()

    print("  Plot saved  → training/outputs/autoencoder_training.png")
    print("  Model saved → saved_models/autoencoder.h5")
    print("  Threshold   → saved_models/autoencoder_threshold.npy")
    print(f"\n  F1-Score: {f1:.3f}")
    print("\nDone ✅")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Autoencoder anomaly detector")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch",  type=int, default=256)
    args = parser.parse_args()
    train(epochs=args.epochs, batch=args.batch)
