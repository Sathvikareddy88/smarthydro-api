"""
models/loader.py
────────────────
Loads all trained ML models at application startup.
"""

from __future__ import annotations
import logging
import os
import numpy as np
from config.settings import settings

logger = logging.getLogger(__name__)

registry: dict[str, object] = {
    "lstm":        None,
    "cnn":         None,
    "yolo":        None,
    "autoencoder": None,
    "rl_policy":   None,
}


def get_model(name: str):
    return registry.get(name)


def _load_lstm():
    """Load the Keras LSTM model for pH / temperature prediction."""
    try:
        import tensorflow as tf
        from tensorflow import keras
        path = settings.LSTM_MODEL_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(f"LSTM model not found at {path}")
        
        # Load without compiling (fixes Keras 3.x compatibility)
        model = keras.models.load_model(path, compile=False)
        
        # Recompile with correct metrics
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss='mae',
            metrics=['mse']
        )
        
        logger.info("✅ LSTM model loaded from %s", path)
        return model
    except Exception as exc:
        logger.warning("⚠️  LSTM model unavailable (%s) — stub will be used.", exc)
        return None


def _load_cnn():
    """Load the ResNet-50 PyTorch model for growth stage classification."""
    try:
        import torch
        path = settings.CNN_MODEL_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(f"CNN model not found at {path}")
        model = torch.load(path, map_location="cpu")
        model.eval()
        logger.info("✅ CNN (ResNet-50) model loaded from %s", path)
        return model
    except Exception as exc:
        logger.warning("⚠️  CNN model unavailable (%s) — stub will be used.", exc)
        return None


def _load_yolo():
    """Load the YOLOv8 pest/disease detection model."""
    try:
        from ultralytics import YOLO
        path = settings.YOLO_MODEL_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(f"YOLO model not found at {path}")
        model = YOLO(path)
        logger.info("✅ YOLOv8 model loaded from %s", path)
        return model
    except Exception as exc:
        logger.warning("⚠️  YOLOv8 model unavailable (%s) — stub will be used.", exc)
        return None


def _load_autoencoder():
    """Load the Keras Autoencoder for operational anomaly detection."""
    try:
        import tensorflow as tf
        from tensorflow import keras
        path = settings.AUTOENCODER_MODEL_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(f"Autoencoder not found at {path}")
        
        # Load without compiling (fixes Keras 3.x compatibility)
        model = keras.models.load_model(path, compile=False)
        
        # Recompile with correct metrics
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss='mse'
        )
        
        logger.info("✅ Autoencoder loaded from %s", path)
        return model
    except Exception as exc:
        logger.warning("⚠️  Autoencoder unavailable (%s) — stub will be used.", exc)
        return None


def _load_rl_policy():
    """Load the Stable-Baselines3 PPO nutrient dosing policy."""
    try:
        from stable_baselines3 import PPO
        path = settings.RL_POLICY_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(f"RL policy not found at {path}")
        model = PPO.load(path)
        logger.info("✅ PPO RL policy loaded from %s", path)
        return model
    except Exception as exc:
        logger.warning("⚠️  PPO policy unavailable (%s) — stub will be used.", exc)
        return None


def load_all():
    """Load all models into the registry."""
    registry["lstm"]        = _load_lstm()
    registry["cnn"]         = _load_cnn()
    registry["yolo"]        = _load_yolo()
    registry["autoencoder"] = _load_autoencoder()
    registry["rl_policy"]   = _load_rl_policy()

    loaded   = [k for k, v in registry.items() if v is not None]
    missing  = [k for k, v in registry.items() if v is None]
    logger.info("Model registry — loaded: %s | stubs: %s", loaded, missing)