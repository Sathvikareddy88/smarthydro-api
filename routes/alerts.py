"""
routes/alerts.py
────────────────
Alert management endpoints.

GET    /alerts           — List alerts (filterable by level, resolved)
GET    /alerts/<id>      — Single alert
PATCH  /alerts/<id>/resolve — Mark alert resolved
DELETE /alerts/<id>      — Remove alert
"""

from flask import Blueprint, request
from bson import ObjectId
from datetime import datetime, timezone
from db.mongo import get_collection
from utils.helpers import serialize_doc, serialize_docs, ok, err

alerts_bp = Blueprint("alerts", __name__, url_prefix="/alerts")


@alerts_bp.get("")
def list_alerts():
    """
    Query params:
      level     — info | warning | critical
      resolved  — true | false
      page      — int (default 1)
      per_page  — int (default 20, max 100)
    """
    page     = max(int(request.args.get("page",     1)), 1)
    per_page = min(int(request.args.get("per_page", 20)), 100)
    level    = request.args.get("level")
    resolved = request.args.get("resolved")

    query = {}
    if level:
        query["level"] = level
    if resolved is not None:
        query["resolved"] = resolved.lower() == "true"

    col   = get_collection("alerts")
    total = col.count_documents(query)
    docs  = list(
        col.find(query)
           .sort("created_at", -1)
           .skip((page - 1) * per_page)
           .limit(per_page)
    )

    return ok({
        "total":    total,
        "page":     page,
        "per_page": per_page,
        "alerts":   serialize_docs(docs),
    })


@alerts_bp.get("/<alert_id>")
def get_alert(alert_id: str):
    try:
        doc = get_collection("alerts").find_one({"_id": ObjectId(alert_id)})
    except Exception:
        return err("Invalid ID format.", 400)
    if not doc:
        return err("Alert not found.", 404)
    return ok(serialize_doc(doc))


@alerts_bp.patch("/<alert_id>/resolve")
def resolve_alert(alert_id: str):
    """Mark an alert as resolved."""
    try:
        result = get_collection("alerts").update_one(
            {"_id": ObjectId(alert_id)},
            {"$set": {"resolved": True, "resolved_at": datetime.now(timezone.utc)}}
        )
    except Exception:
        return err("Invalid ID format.", 400)

    if result.matched_count == 0:
        return err("Alert not found.", 404)
    return ok({"resolved": alert_id})


@alerts_bp.delete("/<alert_id>")
def delete_alert(alert_id: str):
    try:
        result = get_collection("alerts").delete_one({"_id": ObjectId(alert_id)})
    except Exception:
        return err("Invalid ID format.", 400)
    if result.deleted_count == 0:
        return err("Alert not found.", 404)
    return ok({"deleted": alert_id})
