"""
models/inference.py
────────────────────
All ML inference logic lives here.
Each function:
  1. Attempts real model inference using registry[name]
  2. Falls back to a physics-informed deterministic stub
     if the model file is absent (dev / CI / demo mode)

This pattern means the API is always runnable — even without
trained model weights — for development and testing.
"""

from __future__ import annotations
import math
import logging
import numpy as np
from datetime import datetime, timezone
from config.settings import settings
from models.loader import get_model

logger = logging.getLogger(__name__)

GROWTH_STAGE_LABELS = ["seedling", "vegetative", "flowering", "harvest"]
GROWTH_STAGE_MAP    = {s: i for i, s in enumerate(GROWTH_STAGE_LABELS)}

# Lighting profiles per growth stage  [ppfd, photoperiod_h, spectrum_note]
LIGHTING_PROFILES = {
    "seedling":   (200,  16, "Favour blue spectrum (440–490 nm) for compact leaf growth"),
    "vegetative": (400,  18, "Balanced blue/red (1:1) for rapid canopy development"),
    "flowering":  (600,  12, "Increase red (630–660 nm) to stimulate flowering"),
    "harvest":    (300,  14, "Reduce intensity; maintain red to maximise sugar accumulation"),
}

# Detection class → recommended action
PEST_ACTIONS = {
    "aphid":             "Apply neem oil spray; isolate affected tray",
    "whitefly":          "Deploy yellow sticky traps; introduce Encarsia formosa",
    "powdery_mildew":    "Reduce humidity; apply potassium bicarbonate solution",
    "leaf_chlorosis":    "Increase nitrogen in nutrient solution by 10%",
    "tip_burn":          "Increase calcium concentration; improve air circulation",
    "botrytis":          "Remove infected leaves; apply copper-based fungicide",
    "root_rot":          "Increase dissolved oxygen; reduce water temperature",
    "nutrient_deficiency": "Rebalance EC; check pH is within 5.5–6.5 range",
}


# ─── 1. LSTM — pH & Temperature Forecasting ───────────────────────────────────

def predict_ph_temperature(
    window: list[list[float]],
    horizon: int = 2
) -> dict:
    """
    Args:
        window  : list of feature vectors, length = LSTM_LOOKBACK
        horizon : number of 15-min steps to forecast
    Returns:
        dict with ph_forecast, temp_forecast lists
    """
    model = get_model("lstm")

    if model is not None:
        # ── Real inference ──
        import numpy as np
        x = np.array([window], dtype="float32")  # shape (1, T, features)
        preds = model.predict(x, verbose=0)
        # Expect output shape (1, horizon, 2) → [pH, temp] per step
        ph_forecast   = preds[0, :horizon, 0].tolist()
        temp_forecast = preds[0, :horizon, 1].tolist()
    else:
        # ── Stub: sinusoidal drift simulation ──
        last = window[-1] if window else [6.0, 1.4, 22.0, 65.0, 0.07]
        ph_base   = last[0]
        temp_base = last[2]
        ph_forecast   = [round(ph_base   + 0.03 * math.sin(i * 0.8), 3) for i in range(1, horizon + 1)]
        temp_forecast = [round(temp_base + 0.15 * math.cos(i * 0.5), 2) for i in range(1, horizon + 1)]

    ph_alert   = any(p < settings.PH_LOW or p > settings.PH_HIGH for p in ph_forecast)
    temp_alert = any(t < settings.TEMP_LOW or t > settings.TEMP_HIGH for t in temp_forecast)

    return {
        "model":            "lstm",
        "ph_forecast":      ph_forecast,
        "temp_forecast":    temp_forecast,
        "horizon_steps":    horizon,
        "minutes_per_step": 15,
        "ph_alert":         ph_alert,
        "temp_alert":       temp_alert,
        "created_at":       datetime.now(timezone.utc).isoformat(),
    }


# ─── 2. PPO RL — Nutrient Dosing Decision ─────────────────────────────────────

def decide_nutrient_dose(
    ec_current: float,
    ph_current: float,
    growth_stage: str,
    day_in_cycle: int,
    hour_of_day: int,
) -> dict:
    """
    Run the PPO policy to decide the next nutrient dosing action.
    State vector (must match training env observation space):
      [ec_current, ph_current, stage_enc, day_norm, hour_sin, hour_cos]
    Actions: 0=decrease, 1=maintain, 2=increase
    """
    model = get_model("rl_policy")

    stage_idx  = GROWTH_STAGE_MAP.get(growth_stage, 1)
    ec_target  = settings.EC_SETPOINTS[stage_idx]
    day_norm   = day_in_cycle / 90.0
    hour_sin   = math.sin(2 * math.pi * hour_of_day / 24)
    hour_cos   = math.cos(2 * math.pi * hour_of_day / 24)

    obs = np.array([
        ec_current, ph_current, stage_idx / 3.0,
        day_norm, hour_sin, hour_cos
    ], dtype="float32")

    if model is not None:
        # ── Real PPO inference ──
        action_idx, _ = model.predict(obs, deterministic=True)
        action_map    = {0: "decrease", 1: "maintain", 2: "increase"}
        action        = action_map[int(action_idx)]
        confidence    = 0.92  # SB3 policy confidence not directly exposed; placeholder
    else:
        # ── Stub: rule-based fallback mimicking optimal policy ──
        ec_diff = ec_current - ec_target
        if ec_diff < -0.15:
            action, confidence = "increase", 0.87
        elif ec_diff > 0.15:
            action, confidence = "decrease", 0.85
        else:
            action, confidence = "maintain", 0.91

    ec_delta = {"increase": 0.1, "maintain": 0.0, "decrease": -0.1}[action]

    reasoning = (
        f"Current EC {ec_current:.2f} mS/cm vs. stage target {ec_target:.2f} mS/cm "
        f"({growth_stage}, day {day_in_cycle}). Policy recommends: {action.upper()}."
    )

    return {
        "model":      "ppo_rl",
        "action":     action,
        "ec_target":  ec_target,
        "ec_delta":   ec_delta,
        "confidence": round(confidence, 3),
        "reasoning":  reasoning,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


# ─── 3. ResNet-50 CNN — Growth Stage Classification ───────────────────────────

def classify_growth_stage(image_bytes: bytes) -> dict:
    """
    Classify plant growth stage from image bytes.
    Returns growth stage, lighting recommendation.
    """
    model = get_model("cnn")

    if model is not None:
        import torch
        from torchvision import transforms
        from PIL import Image
        import io

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        img    = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1)[0]
            idx    = int(torch.argmax(probs))
            conf   = float(probs[idx])

        stage = GROWTH_STAGE_LABELS[idx]
    else:
        # ── Stub ──
        stage, conf = "vegetative", 0.88

    ppfd, photoperiod, spectrum = LIGHTING_PROFILES[stage]

    return {
        "model":         "resnet50_cnn",
        "growth_stage":  stage,
        "confidence":    round(conf, 3),
        "light_ppfd":    ppfd,
        "photoperiod_h": photoperiod,
        "spectrum_note": spectrum,
        "created_at":    datetime.now(timezone.utc).isoformat(),
    }


# ─── 4. YOLOv8 — Pest & Disease Detection ────────────────────────────────────

def detect_pests(image_bytes: bytes) -> dict:
    """
    Run YOLOv8 on plant image; return detected classes, bboxes, alert level.
    """
    model = get_model("yolo")

    detections = []

    if model is not None:
        import io
        from PIL import Image

        img     = Image.open(io.BytesIO(image_bytes))
        results = model(img, conf=settings.YOLO_CONF_THRESHOLD, verbose=False)

        for r in results:
            for box in r.boxes:
                cls_name   = model.names[int(box.cls)]
                confidence = float(box.conf)
                x1, y1, x2, y2 = [float(v) / max(r.orig_shape) for v in box.xyxy[0]]
                detections.append({
                    "class_name": cls_name,
                    "confidence": round(confidence, 3),
                    "bbox":       [round(x1,3), round(y1,3), round(x2,3), round(y2,3)],
                    "action":     PEST_ACTIONS.get(cls_name, "Flag for manual review"),
                })
    else:
        # ── Stub: healthy plant (no detections) ──
        detections = []

    # Determine alert level
    if not detections:
        alert_level, summary = "info", "No pests or deficiencies detected. Plant appears healthy."
    elif max(d["confidence"] for d in detections) > 0.7:
        alert_level = "critical"
        classes = list({d["class_name"] for d in detections})
        summary = f"HIGH CONFIDENCE detection: {', '.join(classes)}. Immediate action recommended."
    else:
        alert_level = "warning"
        classes = list({d["class_name"] for d in detections})
        summary = f"Low-confidence detection: {', '.join(classes)}. Monitor closely."

    return {
        "model":       "yolov8",
        "detections":  detections,
        "alert_level": alert_level,
        "summary":     summary,
        "created_at":  datetime.now(timezone.utc).isoformat(),
    }


# ─── 5. Autoencoder — Operational Anomaly Detection ──────────────────────────

def score_anomaly(feature_vector: list[float]) -> dict:
    """
    Compute reconstruction error for a single feature vector.
    Returns is_anomaly flag and alert level.
    """
    model     = get_model("autoencoder")
    threshold = settings.AUTOENCODER_THRESHOLD

    if model is not None:
        import numpy as np
        x      = np.array([feature_vector], dtype="float32")
        recon  = model.predict(x, verbose=0)
        error  = float(np.mean((x - recon) ** 2))
    else:
        # ── Stub: small fixed error (simulate healthy state) ──
        error = 0.012

    is_anomaly = error > threshold

    if is_anomaly and error > threshold * 2:
        alert_level = "critical"
    elif is_anomaly:
        alert_level = "warning"
    else:
        alert_level = "info"

    return {
        "model":                "autoencoder",
        "reconstruction_error": round(error, 6),
        "threshold":            threshold,
        "is_anomaly":           is_anomaly,
        "alert_level":          alert_level,
        "created_at":           datetime.now(timezone.utc).isoformat(),
    }
