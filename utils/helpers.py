"""
utils/helpers.py
────────────────
Shared utility functions:
  - MongoDB ObjectId → str serialization
  - Feature engineering for ML model inputs
  - Alert creation helper
  - Standard API response wrapper
"""

from __future__ import annotations
import json
from datetime import datetime, timezone
from bson import ObjectId
from typing import Any


# ─── Serialization ────────────────────────────────────────────────────────────

class MongoJSONEncoder(json.JSONEncoder):
    """Serialize ObjectId and datetime objects for JSON responses."""
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def serialize_doc(doc: dict) -> dict:
    """Convert a MongoDB document to a JSON-safe dict."""
    if doc is None:
        return {}
    if "_id" in doc:
        doc["_id"] = str(doc["_id"])
    return doc


def serialize_docs(docs: list[dict]) -> list[dict]:
    return [serialize_doc(d) for d in docs]


# ─── API Response Wrappers ────────────────────────────────────────────────────

def ok(data: Any, message: str = "success", status: int = 200):
    """Standard success response."""
    from flask import jsonify
    return jsonify({"status": "ok", "message": message, "data": data}), status


def err(message: str, status: int = 400, details: Any = None):
    """Standard error response."""
    from flask import jsonify
    payload = {"status": "error", "message": message}
    if details:
        payload["details"] = details
    return jsonify(payload), status


# ─── Feature Engineering ─────────────────────────────────────────────────────

GROWTH_STAGE_MAP = {"seedling": 0, "vegetative": 1, "flowering": 2, "harvest": 3}

def reading_to_feature_vector(reading: dict) -> list[float]:
    """
    Convert a raw reading document into a numeric feature vector
    consumable by the Autoencoder and LSTM models.

    Feature order (must match training):
      [ph, ec, temperature, humidity, light_lux,
       growth_stage_encoded, day_in_cycle_norm, hour_sin, hour_cos, nutrient_dose]
    """
    stage = GROWTH_STAGE_MAP.get(reading.get("growth_stage", "vegetative"), 1)
    day_norm = reading.get("day_in_cycle", 0) / 90.0   # normalise to ~0-1 for 90-day cycle
    import math
    hour = reading.get("hour_of_day", 12)
    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)

    return [
        reading.get("ph", 6.0),
        reading.get("ec", 1.4),
        reading.get("temperature", 22.0),
        reading.get("humidity", 65.0),
        reading.get("light_lux", 5000.0) / 65535.0,  # normalise to 0-1
        stage / 3.0,                                   # normalise 0-1
        day_norm,
        hour_sin,
        hour_cos,
        reading.get("nutrient_dose", 0.0) / 100.0,
    ]


def build_lstm_window(docs: list[dict]) -> list[list[float]]:
    """Convert a list of reading documents into an LSTM input window."""
    return [reading_to_feature_vector(d) for d in docs]


# ─── Alert Helper ─────────────────────────────────────────────────────────────

def create_alert(source: str, level: str, message: str) -> dict:
    """Build an alert document ready for MongoDB insertion."""
    return {
        "source":     source,
        "level":      level,
        "message":    message,
        "resolved":   False,
        "created_at": datetime.now(timezone.utc),
    }
