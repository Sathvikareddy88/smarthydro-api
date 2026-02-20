"""
app.py
──────
SmartHydro ML API — Flask application factory.

Startup sequence:
  1. Load environment config
  2. Load all ML models into registry
  3. Connect to MongoDB and ensure indexes
  4. Register all route blueprints
  5. Register error handlers

Run locally:
  python app.py

Production (gunicorn):
  gunicorn -w 2 -b 0.0.0.0:5000 "app:create_app()"
"""

import logging
import os
from flask import Flask, jsonify
from flask_cors import CORS

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("smarthydro")


def create_app() -> Flask:
    app = Flask(__name__)

    # ── Config ────────────────────────────────────────────────────────────────
    from config.settings import settings
    app.config["SECRET_KEY"] = settings.SECRET_KEY
    app.config["DEBUG"]      = settings.FLASK_DEBUG

    # ── CORS ──────────────────────────────────────────────────────────────────
    CORS(app, resources={r"/*": {"origins": "*"}})

    # ── Load ML Models ────────────────────────────────────────────────────────
    from models.loader import load_all
    with app.app_context():
        logger.info("Loading ML models…")
        load_all()

    # ── MongoDB Indexes ───────────────────────────────────────────────────────
    from db.mongo import ensure_indexes
    with app.app_context():
        try:
            ensure_indexes()
        except Exception as exc:
            logger.warning("MongoDB index setup failed (DB unreachable?): %s", exc)

    # ── Blueprints ────────────────────────────────────────────────────────────
    from routes.readings  import readings_bp
    from routes.predict   import predict_bp
    from routes.alerts    import alerts_bp
    from routes.dashboard import dashboard_bp

    app.register_blueprint(readings_bp)
    app.register_blueprint(predict_bp)
    app.register_blueprint(alerts_bp)
    app.register_blueprint(dashboard_bp)

    # ── Root health check ─────────────────────────────────────────────────────
    @app.get("/")
    def health():
        return jsonify({
            "service": "SmartHydro ML API",
            "version": "1.0.0",
            "status":  "running",
            "endpoints": [
                "POST /readings",
                "GET  /readings",
                "GET  /readings/window",
                "POST /predict/lstm",
                "POST /predict/dose",
                "POST /predict/growth",
                "POST /predict/pests",
                "POST /predict/anomaly",
                "GET  /predict/health",
                "GET  /alerts",
                "PATCH /alerts/<id>/resolve",
                "GET  /dashboard/summary",
                "GET  /dashboard/trends",
                "GET  /dashboard/dosing-log",
                "GET  /dashboard/growth-log",
            ]
        })

    # ── Error Handlers ────────────────────────────────────────────────────────
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"status": "error", "message": "Endpoint not found."}), 404

    @app.errorhandler(405)
    def method_not_allowed(e):
        return jsonify({"status": "error", "message": "Method not allowed."}), 405

    @app.errorhandler(500)
    def server_error(e):
        logger.exception("Unhandled 500 error")
        return jsonify({"status": "error", "message": "Internal server error."}), 500

    logger.info("✅ SmartHydro API ready.")
    return app


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app  = create_app()
    app.run(host="0.0.0.0", port=port, debug=True)
