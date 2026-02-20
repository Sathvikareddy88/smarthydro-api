"""
config/settings.py
──────────────────
Loads all environment variables and exposes typed settings
to the rest of the application.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Flask
    FLASK_ENV: str        = os.getenv("FLASK_ENV", "production")
    FLASK_DEBUG: bool     = os.getenv("FLASK_DEBUG", "0") == "1"
    SECRET_KEY: str       = os.getenv("SECRET_KEY", "change-me")

    # MongoDB
    MONGO_URI: str        = os.getenv("MONGO_URI", "mongodb://localhost:27017/smarthydro")
    DB_NAME: str          = os.getenv("DB_NAME", "smarthydro")
    COL_READINGS: str     = os.getenv("COLLECTION_READINGS",    "readings")
    COL_PREDICTIONS: str  = os.getenv("COLLECTION_PREDICTIONS", "predictions")
    COL_DOSING: str       = os.getenv("COLLECTION_DOSING_LOGS", "dosing_logs")
    COL_ALERTS: str       = os.getenv("COLLECTION_ALERTS",      "alerts")
    COL_GROWTH: str       = os.getenv("COLLECTION_GROWTH",      "growth_logs")

    # Model paths
    LSTM_MODEL_PATH:        str = os.getenv("LSTM_MODEL_PATH",        "saved_models/lstm_ph_temp.h5")
    CNN_MODEL_PATH:         str = os.getenv("CNN_MODEL_PATH",         "saved_models/resnet50_growth.pt")
    YOLO_MODEL_PATH:        str = os.getenv("YOLO_MODEL_PATH",        "saved_models/yolov8_pest.pt")
    AUTOENCODER_MODEL_PATH: str = os.getenv("AUTOENCODER_MODEL_PATH", "saved_models/autoencoder.h5")
    RL_POLICY_PATH:         str = os.getenv("RL_POLICY_PATH",         "saved_models/ppo_dosing_policy.zip")

    # Inference
    LSTM_LOOKBACK:          int   = int(os.getenv("LSTM_LOOKBACK", "24"))
    AUTOENCODER_THRESHOLD:  float = float(os.getenv("AUTOENCODER_THRESHOLD", "0.045"))
    YOLO_CONF_THRESHOLD:    float = float(os.getenv("YOLO_CONF_THRESHOLD", "0.5"))

    # Crop setpoints
    EC_SETPOINTS:           list  = [float(x) for x in os.getenv("EC_SETPOINTS", "0.8,1.4,1.8,1.6").split(",")]
    PH_LOW:                 float = float(os.getenv("PH_OPTIMAL_LOW",  "5.5"))
    PH_HIGH:                float = float(os.getenv("PH_OPTIMAL_HIGH", "6.5"))
    TEMP_LOW:               float = float(os.getenv("TEMP_OPTIMAL_LOW",  "18.0"))
    TEMP_HIGH:              float = float(os.getenv("TEMP_OPTIMAL_HIGH", "30.0"))


settings = Settings()
