"""
training/data_generators/hydro_data.py
────────────────────────────────────────
Generates realistic synthetic hydroponic time-series data for
training all ML models when real labelled data is not yet available.

Simulates:
  - pH drift with circadian cycle + plant metabolism noise
  - EC depletion as plants uptake nutrients
  - Temperature oscillation (day/night cycle)
  - Humidity variation
  - Light intensity (12–18h photoperiod)
  - Growth stage progression over a 90-day crop cycle
  - Anomaly injection (pump failures, pH spikes)
  - Disease / pest symptom injection (for YOLOv8 labels)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG = np.random.default_rng(42)

GROWTH_STAGES   = ["seedling", "vegetative", "flowering", "harvest"]
STAGE_DAYS      = [0, 14, 42, 72]          # stage starts at day N
EC_TARGETS      = [0.8, 1.4, 1.8, 1.6]    # mS/cm per stage
PPFD_TARGETS    = [200, 400, 600, 300]     # µmol/m²/s per stage
PHOTOPERIODS    = [16, 18, 12, 14]         # hours of light per stage


def _stage_at_day(day: int) -> int:
    for i in reversed(range(len(STAGE_DAYS))):
        if day >= STAGE_DAYS[i]:
            return i
    return 0


def generate_time_series(
    n_days: int = 90,
    interval_min: int = 15,
    crop_type: str = "lettuce",
    inject_anomalies: bool = True,
) -> pd.DataFrame:
    """
    Generate a full crop-cycle time-series DataFrame.

    Returns columns:
      timestamp, ph, ec, temperature, humidity, light_lux,
      growth_stage, day_in_cycle, hour_of_day, nutrient_dose,
      crop_type, is_anomaly
    """
    steps = int(n_days * 24 * 60 / interval_min)
    start = datetime(2024, 1, 1)
    rows  = []

    ph   = 6.0
    ec   = 1.0
    temp = 22.0

    # Anomaly windows (start_step, duration_steps)
    anomaly_windows = set()
    if inject_anomalies:
        for _ in range(int(n_days / 7)):  # ~1 anomaly per week
            s = RNG.integers(50, steps - 50)
            for k in range(RNG.integers(2, 8)):
                anomaly_windows.add(s + k)

    for i in range(steps):
        ts       = start + timedelta(minutes=i * interval_min)
        day      = i * interval_min // (24 * 60)
        hour     = ts.hour + ts.minute / 60
        stage_i  = _stage_at_day(day)
        stage    = GROWTH_STAGES[stage_i]
        photoperiod = PHOTOPERIODS[stage_i]

        # ── pH: circadian drift + noise ──
        ph_drift = 0.002 * np.sin(2 * np.pi * hour / 24)
        ph      += ph_drift + RNG.normal(0, 0.005)
        ph       = np.clip(ph, 4.5, 8.5)

        # ── EC: slow depletion, replenished by dosing ──
        ec_target = EC_TARGETS[stage_i]
        ec       -= RNG.uniform(0.001, 0.003)   # uptake
        dose      = 0.0
        if ec < ec_target - 0.1:
            dose = RNG.uniform(3, 8)
            ec  += 0.15

        # ── Temperature: day/night ──
        temp_target = 22.0 + 3.0 * np.sin(2 * np.pi * (hour - 6) / 24)
        temp       += 0.1 * (temp_target - temp) + RNG.normal(0, 0.1)

        # ── Humidity ──
        humidity = 65 + 10 * np.sin(2 * np.pi * hour / 24) + RNG.normal(0, 2)
        humidity = np.clip(humidity, 30, 95)

        # ── Light: photoperiod on/off ──
        light_on = (6 <= hour < 6 + photoperiod)
        if light_on:
            ppfd      = PPFD_TARGETS[stage_i]
            light_lux = ppfd * 54 + RNG.normal(0, 200)   # rough lux conversion
        else:
            light_lux = RNG.uniform(0, 50)

        # ── Anomaly injection ──
        is_anomaly = i in anomaly_windows
        if is_anomaly:
            anomaly_type = RNG.choice(["ph_spike", "ec_drop", "temp_surge"])
            if anomaly_type == "ph_spike":
                ph   += RNG.uniform(0.5, 1.5)
            elif anomaly_type == "ec_drop":
                ec   -= RNG.uniform(0.3, 0.6)
            else:
                temp += RNG.uniform(4, 8)

        rows.append({
            "timestamp":    ts,
            "ph":           round(float(ph),       3),
            "ec":           round(float(ec),        3),
            "temperature":  round(float(temp),      2),
            "humidity":     round(float(humidity),  1),
            "light_lux":    round(max(0, float(light_lux)), 1),
            "growth_stage": stage,
            "stage_idx":    stage_i,
            "day_in_cycle": int(day),
            "hour_of_day":  int(ts.hour),
            "nutrient_dose": round(float(dose),    2),
            "crop_type":    crop_type,
            "is_anomaly":   bool(is_anomaly),
        })

    return pd.DataFrame(rows)


def generate_multi_crop(
    crops: list[str] = ["lettuce", "spinach", "basil"],
    n_days: int = 90,
) -> pd.DataFrame:
    """Concatenate time-series for multiple crop types."""
    frames = [generate_time_series(n_days=n_days, crop_type=c) for c in crops]
    df     = pd.concat(frames, ignore_index=True)
    df     = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def build_lstm_sequences(
    df: pd.DataFrame,
    lookback: int = 24,
    horizon: int  = 2,
    features: list[str] | None = None,
    targets: list[str]  | None = None,
) -> tuple:
    """
    Slide a window over the time-series to produce (X, y) tensors for LSTM.

    X shape: (N, lookback, n_features)
    y shape: (N, horizon, n_targets)
    """
    import numpy as np

    features = features or ["ph", "ec", "temperature", "humidity",
                             "light_lux_norm", "stage_norm",
                             "day_norm", "hour_sin", "hour_cos", "nutrient_dose_norm"]
    targets  = targets  or ["ph", "temperature"]

    df = df.copy().sort_values("timestamp").reset_index(drop=True)

    # ── Feature engineering ──
    df["light_lux_norm"]    = df["light_lux"] / 65535
    df["stage_norm"]         = df["stage_idx"] / 3
    df["day_norm"]           = df["day_in_cycle"] / 90
    df["hour_sin"]           = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"]           = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["nutrient_dose_norm"] = df["nutrient_dose"] / 100

    feat_arr = df[features].values.astype("float32")
    tgt_arr  = df[targets].values.astype("float32")

    X, y = [], []
    for i in range(lookback, len(df) - horizon):
        X.append(feat_arr[i - lookback : i])
        y.append(tgt_arr[i : i + horizon])

    return np.array(X), np.array(y)


def build_autoencoder_dataset(df: pd.DataFrame) -> tuple:
    """
    Build (X_normal, X_anomaly) arrays for autoencoder training.
    Normal: is_anomaly == False, Anomaly: is_anomaly == True
    """
    import numpy as np

    feature_cols = ["ph", "ec", "temperature", "humidity",
                    "light_lux", "stage_idx", "day_in_cycle",
                    "hour_of_day", "nutrient_dose"]

    df = df.copy()
    df["light_lux"] = df["light_lux"] / 65535
    df["stage_idx"] = df["stage_idx"] / 3
    df["day_in_cycle"] = df["day_in_cycle"] / 90

    # Normalise with hour encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)

    cols = ["ph", "ec", "temperature", "humidity", "light_lux",
            "stage_idx", "day_in_cycle", "hour_sin", "hour_cos", "nutrient_dose"]

    normal  = df[~df["is_anomaly"]][cols].values.astype("float32")
    anomaly = df[ df["is_anomaly"]][cols].values.astype("float32")
    return normal, anomaly


def build_rl_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a structured dataset for RL environment simulation.
    Includes state transitions and optimal action labels.
    """
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    df["ec_target"] = df["stage_idx"].map({i: v for i, v in enumerate(EC_TARGETS)})
    df["ec_delta"]  = df["ec"].diff().fillna(0)
    df["action"]    = 1   # maintain by default
    df.loc[df["ec"] < df["ec_target"] - 0.15, "action"] = 2   # increase
    df.loc[df["ec"] > df["ec_target"] + 0.15, "action"] = 0   # decrease
    return df


if __name__ == "__main__":
    print("Generating synthetic dataset…")
    df = generate_multi_crop()
    print(df.head())
    print(f"Total rows: {len(df):,} | Anomalies: {df['is_anomaly'].sum():,}")
    df.to_csv("synthetic_hydro_data.csv", index=False)
    print("Saved to synthetic_hydro_data.csv")
