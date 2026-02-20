"""
routes/readings.py
──────────────────
/readings  — Store and retrieve operational readings (feature store).

POST /readings          — Ingest a new reading; also runs anomaly check
GET  /readings          — Paginated list (latest first)
GET  /readings/<id>     — Single reading by ID
GET  /readings/window   — Latest N readings as LSTM feature window
DELETE /readings/<id>   — Remove a reading
"""

from flask import Blueprint, request
from bson import ObjectId
from datetime import datetime, timezone
from db.mongo import get_collection
from models.inference import score_anomaly
from utils.helpers import (
    serialize_doc, serialize_docs, ok, err,
    reading_to_feature_vector, build_lstm_window, create_alert
)
from config.settings import settings

readings_bp = Blueprint("readings", __name__, url_prefix="/readings")


@readings_bp.post("")
def create_reading():
    """
    Ingest a new operational reading.
    After storing, automatically runs the Autoencoder anomaly check.
    If anomalous, an alert document is also created.
    """
    data = request.get_json(silent=True)
    if not data:
        return err("Request body must be valid JSON.", 400)

    # Add server-side timestamp if not provided
    data.setdefault("timestamp", datetime.now(timezone.utc))

    col = get_collection("readings")
    result = col.insert_one(data)
    reading_id = str(result.inserted_id)

    # ── Auto anomaly check ──
    try:
        fv      = reading_to_feature_vector(data)
        anomaly = score_anomaly(fv)

        if anomaly["is_anomaly"]:
            alert = create_alert(
                source  = "autoencoder",
                level   = anomaly["alert_level"],
                message = (
                    f"Anomaly detected for reading {reading_id}. "
                    f"Reconstruction error: {anomaly['reconstruction_error']} "
                    f"(threshold: {anomaly['threshold']})"
                )
            )
            get_collection("alerts").insert_one(alert)
    except Exception as exc:
        # Non-fatal — don't fail the ingestion
        pass

    return ok({"id": reading_id}, "Reading stored.", 201)


@readings_bp.get("")
def list_readings():
    """
    Return paginated readings (newest first).
    Query params: page (int), per_page (int), crop_type (str)
    """
    page     = max(int(request.args.get("page",     1)), 1)
    per_page = min(int(request.args.get("per_page", 20)), 100)
    crop     = request.args.get("crop_type")

    query = {}
    if crop:
        query["crop_type"] = crop

    col   = get_collection("readings")
    total = col.count_documents(query)
    docs  = list(
        col.find(query)
           .sort("timestamp", -1)
           .skip((page - 1) * per_page)
           .limit(per_page)
    )

    return ok({
        "total":    total,
        "page":     page,
        "per_page": per_page,
        "readings": serialize_docs(docs),
    })


@readings_bp.get("/<reading_id>")
def get_reading(reading_id: str):
    try:
        doc = get_collection("readings").find_one({"_id": ObjectId(reading_id)})
    except Exception:
        return err("Invalid ID format.", 400)

    if not doc:
        return err("Reading not found.", 404)
    return ok(serialize_doc(doc))


@readings_bp.get("/window")
def get_feature_window():
    """
    Return the latest N readings as a feature window for LSTM inference.
    Query param: n (default: LSTM_LOOKBACK from settings)
    """
    n    = int(request.args.get("n", settings.LSTM_LOOKBACK))
    crop = request.args.get("crop_type")

    query = {}
    if crop:
        query["crop_type"] = crop

    docs   = list(
        get_collection("readings")
        .find(query)
        .sort("timestamp", -1)
        .limit(n)
    )
    docs.reverse()  # oldest first for LSTM

    window = build_lstm_window(docs)
    return ok({
        "count":   len(window),
        "window":  window,
        "raw_docs": serialize_docs(docs),
    })


@readings_bp.delete("/<reading_id>")
def delete_reading(reading_id: str):
    try:
        result = get_collection("readings").delete_one({"_id": ObjectId(reading_id)})
    except Exception:
        return err("Invalid ID format.", 400)

    if result.deleted_count == 0:
        return err("Reading not found.", 404)
    return ok({"deleted": reading_id})
