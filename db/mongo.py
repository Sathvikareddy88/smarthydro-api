"""
db/mongo.py
───────────
MongoDB client singleton and collection accessors.
All DB interaction in the app goes through get_collection().

Usage:
    from db.mongo import get_collection
    col = get_collection("readings")
    col.insert_one({...})
"""

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.collection import Collection
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

_client: MongoClient | None = None


def get_client() -> MongoClient:
    """Return the MongoClient singleton, creating it on first call."""
    global _client
    if _client is None:
        _client = MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=5000)
        try:
            _client.admin.command("ping")
            logger.info("✅ MongoDB connected: %s", settings.MONGO_URI)
        except Exception as exc:
            logger.error("❌ MongoDB connection failed: %s", exc)
            raise
    return _client


def get_db():
    return get_client()[settings.DB_NAME]


def get_collection(name: str) -> Collection:
    """Get a collection by its logical name (config key) or literal name."""
    col_map = {
        "readings":    settings.COL_READINGS,
        "predictions": settings.COL_PREDICTIONS,
        "dosing":      settings.COL_DOSING,
        "alerts":      settings.COL_ALERTS,
        "growth":      settings.COL_GROWTH,
    }
    return get_db()[col_map.get(name, name)]


def ensure_indexes():
    """
    Create indexes for commonly queried fields.
    Call once at app startup.
    """
    get_collection("readings").create_index(
        [("timestamp", DESCENDING)], name="idx_readings_ts"
    )
    get_collection("predictions").create_index(
        [("created_at", DESCENDING), ("model", ASCENDING)], name="idx_pred_model_ts"
    )
    get_collection("alerts").create_index(
        [("level", ASCENDING), ("resolved", ASCENDING)], name="idx_alert_level"
    )
    get_collection("dosing").create_index(
        [("timestamp", DESCENDING)], name="idx_dosing_ts"
    )
    logger.info("✅ MongoDB indexes ensured.")
