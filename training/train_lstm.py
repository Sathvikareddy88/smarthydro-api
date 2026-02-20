"""
training/train_lstm.py
───────────────────────
Trains the stacked LSTM model to forecast pH and temperature
2 steps (30 minutes) ahead from a 24-step (6 hour) input window.

Architecture:
  Input  → (batch, 24, 10 features)
  LSTM(128, return_sequences=True)
  Dropout(0.2)
  LSTM(64)
  Dropout(0.2)
  Dense(2 * horizon)  → reshape to (horizon, 2)

Output saved to:  saved_models/lstm_ph_temp.h5

Run:
  python training/train_lstm.py [--epochs 50] [--batch 64] [--lookback 24] [--horizon 2]
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

os.makedirs("saved_models", exist_ok=True)
os.makedirs("training/outputs", exist_ok=True)


def build_model(lookback: int, n_features: int, horizon: int, n_targets: int):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    inp  = keras.Input(shape=(lookback, n_features), name="input")
    x    = layers.LSTM(128, return_sequences=True, name="lstm_1")(inp)
    x    = layers.Dropout(0.2)(x)
    x    = layers.LSTM(64, name="lstm_2")(x)
    x    = layers.Dropout(0.2)(x)
    out  = layers.Dense(horizon * n_targets, name="dense")(x)
    out  = layers.Reshape((horizon, n_targets), name="output")(out)

    model = keras.Model(inp, out)
    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=1e-3),
        loss      = "mae",
        metrics   = ["mse"],
    )
    return model


def train(
    epochs:   int = 50,
    batch:    int = 64,
    lookback: int = 24,
    horizon:  int = 2,
):
    print("=" * 60)
    print("  SmartHydro — LSTM pH/Temperature Trainer")
    print("=" * 60)

    # ── 1. Generate data ──────────────────────────────────────────
    from training.data_generators.hydro_data import (
        generate_multi_crop, build_lstm_sequences
    )
    print("\n[1/5] Generating synthetic time-series data…")
    df = generate_multi_crop(n_days=90)
    print(f"      Rows: {len(df):,}")

    # ── 2. Build sequences ────────────────────────────────────────
    print("[2/5] Building LSTM sequences…")
    X, y = build_lstm_sequences(df, lookback=lookback, horizon=horizon)
    # y shape: (N, horizon, 2)  [ph, temp]
    print(f"      X: {X.shape}  y: {y.shape}")

    # ── 3. Scale ──────────────────────────────────────────────────
    print("[3/5] Scaling features…")
    n, t, f = X.shape
    X_flat  = X.reshape(n * t, f)
    scaler  = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_flat).reshape(n, t, f).astype("float32")
    joblib.dump(scaler, "saved_models/lstm_scaler.pkl")

    # Scale targets separately
    ph_min,   ph_max   = 4.0, 9.0
    temp_min, temp_max = 10.0, 40.0
    y_scaled = y.copy().astype("float32")
    y_scaled[..., 0]   = (y[..., 0] - ph_min)   / (ph_max   - ph_min)
    y_scaled[..., 1]   = (y[..., 1] - temp_min) / (temp_max - temp_min)

    # ── 4. Split ──────────────────────────────────────────────────
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_scaled, y_scaled, test_size=0.2, shuffle=False
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_val, y_val, test_size=0.5, shuffle=False
    )
    print(f"      Train: {len(X_tr):,}  Val: {len(X_val):,}  Test: {len(X_te):,}")

    # ── 5. Train ──────────────────────────────────────────────────
    import tensorflow as tf
    from tensorflow import keras

    print(f"[4/5] Training ({epochs} epochs, batch={batch})…")
    model = build_model(lookback, f, horizon, n_targets=2)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor="val_loss"),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-5),
        keras.callbacks.ModelCheckpoint("saved_models/lstm_ph_temp.h5", save_best_only=True),
    ]

    history = model.fit(
        X_tr, y_tr,
        validation_data = (X_val, y_val),
        epochs          = epochs,
        batch_size      = batch,
        callbacks       = callbacks,
        verbose         = 1,
    )

    # ── 6. Evaluate ───────────────────────────────────────────────
    print("\n[5/5] Evaluating on test set…")
    y_pred    = model.predict(X_te, verbose=0)

    # Inverse-scale for interpretable MAE
    ph_mae    = np.mean(np.abs(
        y_pred[..., 0] * (ph_max - ph_min) + ph_min -
        (y_te[...,  0] * (ph_max - ph_min) + ph_min)
    ))
    temp_mae  = np.mean(np.abs(
        y_pred[..., 1] * (temp_max - temp_min) + temp_min -
        (y_te[...,  1] * (temp_max - temp_min) + temp_min)
    ))
    print(f"\n  pH   MAE: {ph_mae:.4f}")
    print(f"  Temp MAE: {temp_mae:.4f} °C")

    # ── 7. Loss plot ──────────────────────────────────────────────
    plt.figure(figsize=(10, 4))
    plt.plot(history.history["loss"],     label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("MAE"); plt.title("LSTM Training Loss")
    plt.legend(); plt.tight_layout()
    plt.savefig("training/outputs/lstm_loss.png", dpi=150)
    plt.close()
    print("\n  Loss plot saved → training/outputs/lstm_loss.png")
    print("  Model saved     → saved_models/lstm_ph_temp.h5")
    print("\nDone ✅")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM pH/Temp forecaster")
    parser.add_argument("--epochs",   type=int, default=50)
    parser.add_argument("--batch",    type=int, default=64)
    parser.add_argument("--lookback", type=int, default=24)
    parser.add_argument("--horizon",  type=int, default=2)
    args = parser.parse_args()
    train(epochs=args.epochs, batch=args.batch,
          lookback=args.lookback, horizon=args.horizon)
