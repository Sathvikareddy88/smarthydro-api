"""
routes/dashboard.py
────────────────────
Aggregated analytics endpoints consumed by the React frontend.

GET /dashboard/summary      — Latest readings + active alert counts
GET /dashboard/trends       — Time-series data for charts (pH, EC, temp)
GET /dashboard/dosing-log   — Recent RL dosing decisions
GET /dashboard/growth-log   — Recent CNN growth stage classifications
"""

from flask import Blueprint, request
from datetime import datetime, timezone, timedelta
from db.mongo import get_collection
from utils.helpers import serialize_docs, ok, err

dashboard_bp = Blueprint("dashboard", __name__, url_prefix="/dashboard")


@dashboard_bp.get("/summary")
def summary():
    """
    Returns the most recent reading + unresolved alert breakdown by level.
    Used to populate the top status bar of the React dashboard.
    """
    latest_reading = get_collection("readings").find_one(
        {}, sort=[("timestamp", -1)]
    )

    # Alert counts by level
    alert_col    = get_collection("alerts")
    alert_counts = {
        "critical": alert_col.count_documents({"level": "critical", "resolved": False}),
        "warning":  alert_col.count_documents({"level": "warning",  "resolved": False}),
        "info":     alert_col.count_documents({"level": "info",     "resolved": False}),
    }

    # Latest dosing action
    latest_dose = get_collection("dosing").find_one(
        {}, sort=[("created_at", -1)]
    )

    # Latest growth stage
    latest_growth = get_collection("growth").find_one(
        {}, sort=[("created_at", -1)]
    )

    def safe(doc):
        if not doc:
            return None
        doc["_id"] = str(doc["_id"])
        return doc

    return ok({
        "latest_reading":  safe(latest_reading),
        "alert_counts":    alert_counts,
        "latest_dose":     safe(latest_dose),
        "latest_growth":   safe(latest_growth),
    })


@dashboard_bp.get("/trends")
def trends():
    """
    Returns time-series arrays for pH, EC, temperature, humidity, light
    over the last N hours (default 24).

    Query params:
      hours     — int, lookback window in hours (default 24, max 168)
      interval  — "raw" | "hourly" (default "raw")
    """
    hours    = min(int(request.args.get("hours", 24)), 168)
    since    = datetime.now(timezone.utc) - timedelta(hours=hours)

    docs = list(
        get_collection("readings")
        .find({"timestamp": {"$gte": since}})
        .sort("timestamp", 1)
        .limit(500)
    )

    def extract(field):
        return [
            {"t": str(d.get("timestamp", "")), "v": d.get(field)}
            for d in docs if field in d
        ]

    return ok({
        "ph":          extract("ph"),
        "ec":          extract("ec"),
        "temperature": extract("temperature"),
        "humidity":    extract("humidity"),
        "light_lux":   extract("light_lux"),
        "count":       len(docs),
        "since":       since.isoformat(),
    })


@dashboard_bp.get("/dosing-log")
def dosing_log():
    """Last N dosing decisions from the RL agent."""
    n    = min(int(request.args.get("n", 50)), 200)
    docs = list(
        get_collection("dosing")
        .find({})
        .sort("created_at", -1)
        .limit(n)
    )
    return ok({"dosing_log": serialize_docs(docs), "count": len(docs)})


@dashboard_bp.get("/growth-log")
def growth_log():
    """Last N growth stage classifications from the CNN."""
    n    = min(int(request.args.get("n", 50)), 200)
    docs = list(
        get_collection("growth")
        .find({})
        .sort("created_at", -1)
        .limit(n)
    )
    return ok({"growth_log": serialize_docs(docs), "count": len(docs)})
