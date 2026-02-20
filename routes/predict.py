"""
routes/predict.py
──────────────────
All ML inference endpoints.

POST /predict/lstm          — pH & temperature forecast (LSTM)
POST /predict/dose          — Nutrient dosing action (PPO RL agent)
POST /predict/growth        — Growth stage classification (CNN) — accepts image upload
POST /predict/pests         — Pest & disease detection (YOLOv8) — accepts image upload
POST /predict/anomaly       — Anomaly scoring (Autoencoder)
GET  /predict/health        — Model registry health check
"""

from flask import Blueprint, request
from datetime import datetime, timezone
from db.mongo import get_collection
from models.inference import (
    predict_ph_temperature,
    decide_nutrient_dose,
    classify_growth_stage,
    detect_pests,
    score_anomaly,
)
from utils.helpers import (
    ok, err, serialize_doc, create_alert,
    build_lstm_window, reading_to_feature_vector
)
from models.loader import registry
from config.settings import settings

predict_bp = Blueprint("predict", __name__, url_prefix="/predict")


# ─── LSTM: pH & Temperature Forecast ─────────────────────────────────────────

@predict_bp.post("/lstm")
def lstm_predict():
    """
    Forecast pH and temperature for the next N * 15-minute intervals.

    Body (JSON):
      {
        "use_db":   true,          # pull latest window from MongoDB
        "lookback": 24,            # number of past steps to use
        "horizon":  2,             # steps ahead to forecast
        "window":   [[...], ...]   # optional: supply window directly
      }
    """
    body     = request.get_json(silent=True) or {}
    horizon  = int(body.get("horizon",  2))
    lookback = int(body.get("lookback", settings.LSTM_LOOKBACK))
    use_db   = body.get("use_db", True)

    if use_db or not body.get("window"):
        docs = list(
            get_collection("readings")
            .find({})
            .sort("timestamp", -1)
            .limit(lookback)
        )
        docs.reverse()
        window = build_lstm_window(docs)
    else:
        window = body["window"]

    if len(window) < 2:
        return err("Insufficient data. Need at least 2 readings in the window.", 422)

    result = predict_ph_temperature(window, horizon=horizon)

    # Persist prediction
    get_collection("predictions").insert_one({**result, "model": "lstm"})

    # Auto-alert on pH or temp excursion forecast
    if result["ph_alert"]:
        get_collection("alerts").insert_one(create_alert(
            source  = "lstm_ph",
            level   = "warning",
            message = f"pH predicted to leave safe range [5.5–6.5]: {result['ph_forecast']}"
        ))
    if result["temp_alert"]:
        get_collection("alerts").insert_one(create_alert(
            source  = "lstm_temp",
            level   = "warning",
            message = f"Temperature forecast outside optimal range: {result['temp_forecast']}"
        ))

    return ok(result)


# ─── PPO RL: Nutrient Dosing Decision ────────────────────────────────────────

@predict_bp.post("/dose")
def dose_predict():
    """
    Get the RL agent's recommended nutrient dosing action.

    Body (JSON):
      {
        "ec_current":   1.2,
        "ph_current":   6.1,
        "growth_stage": "vegetative",
        "day_in_cycle": 14,
        "hour_of_day":  10,
        "crop_type":    "lettuce"
      }
    """
    body = request.get_json(silent=True) or {}

    required = ["ec_current", "ph_current", "growth_stage", "day_in_cycle", "hour_of_day"]
    missing  = [k for k in required if k not in body]
    if missing:
        return err(f"Missing required fields: {missing}", 400)

    result = decide_nutrient_dose(
        ec_current   = float(body["ec_current"]),
        ph_current   = float(body["ph_current"]),
        growth_stage = str(body["growth_stage"]),
        day_in_cycle = int(body["day_in_cycle"]),
        hour_of_day  = int(body["hour_of_day"]),
    )

    # Log dosing action
    get_collection("dosing").insert_one({
        **result,
        "crop_type": body.get("crop_type", "lettuce"),
    })

    return ok(result)


# ─── CNN: Growth Stage Classification ────────────────────────────────────────

@predict_bp.post("/growth")
def growth_predict():
    """
    Classify plant growth stage from an uploaded image.
    Accepts: multipart/form-data with field 'image'
    """
    if "image" not in request.files:
        return err("No image file provided. Send as multipart/form-data with key 'image'.", 400)

    img_bytes = request.files["image"].read()
    if not img_bytes:
        return err("Empty image file.", 400)

    result = classify_growth_stage(img_bytes)

    # Persist
    get_collection("growth").insert_one({**result})

    return ok(result)


# ─── YOLOv8: Pest & Disease Detection ────────────────────────────────────────

@predict_bp.post("/pests")
def pest_predict():
    """
    Detect pests and diseases from an uploaded plant image.
    Accepts: multipart/form-data with field 'image'
    """
    if "image" not in request.files:
        return err("No image file provided. Send as multipart/form-data with key 'image'.", 400)

    img_bytes = request.files["image"].read()
    if not img_bytes:
        return err("Empty image file.", 400)

    result = detect_pests(img_bytes)

    # Persist + alert if needed
    get_collection("predictions").insert_one({**result, "model": "yolov8"})

    if result["alert_level"] in ("warning", "critical"):
        get_collection("alerts").insert_one(create_alert(
            source  = "yolov8",
            level   = result["alert_level"],
            message = result["summary"],
        ))

    return ok(result)


# ─── Autoencoder: Anomaly Scoring ────────────────────────────────────────────

@predict_bp.post("/anomaly")
def anomaly_predict():
    """
    Score a single feature vector for operational anomalies.

    Body (JSON):
      {
        "features": [6.0, 1.4, 22.0, 65.0, 0.07, 0.33, 0.15, 0.96, 0.25, 0.01]
      }

    Or omit 'features' to auto-build from the latest reading in MongoDB.
    """
    body = request.get_json(silent=True) or {}

    if "features" in body:
        features = body["features"]
    else:
        # Auto-pull latest reading
        doc = get_collection("readings").find_one({}, sort=[("timestamp", -1)])
        if not doc:
            return err("No readings found in DB. Provide 'features' directly.", 422)
        features = reading_to_feature_vector(doc)

    result = score_anomaly(features)

    # Persist + alert
    get_collection("predictions").insert_one({**result, "model": "autoencoder"})

    if result["is_anomaly"]:
        get_collection("alerts").insert_one(create_alert(
            source  = "autoencoder",
            level   = result["alert_level"],
            message = (
                f"Operational anomaly detected. "
                f"Reconstruction error {result['reconstruction_error']} "
                f"exceeds threshold {result['threshold']}."
            )
        ))

    return ok(result)


# ─── Model Registry Health Check ─────────────────────────────────────────────

@predict_bp.get("/health")
def model_health():
    """Return load status for all ML models."""
    status = {
        name: ("loaded" if model is not None else "stub (model file not found)")
        for name, model in registry.items()
    }
    return ok({"models": status})
