"""
utils/schemas.py
────────────────
Pydantic models used for request validation and response serialization
across all API routes.
"""

from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, field_validator


# ─── Enums ────────────────────────────────────────────────────────────────────

class GrowthStage(str, Enum):
    SEEDLING   = "seedling"
    VEGETATIVE = "vegetative"
    FLOWERING  = "flowering"
    HARVEST    = "harvest"

class DosingAction(str, Enum):
    INCREASE = "increase"
    MAINTAIN = "maintain"
    DECREASE = "decrease"

class AlertLevel(str, Enum):
    INFO    = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# ─── Reading (feature input) ───────────────────────────────────────────────────

class ReadingIn(BaseModel):
    """
    A single timestamped operational reading submitted to the /readings endpoint.
    These are stored as historical features and consumed by LSTM / Autoencoder.
    """
    timestamp:     datetime = Field(default_factory=datetime.utcnow)
    ph:            float    = Field(..., ge=0.0,   le=14.0,  description="Inferred or measured pH value")
    ec:            float    = Field(..., ge=0.0,   le=5.0,   description="Electrical conductivity (mS/cm)")
    temperature:   float    = Field(..., ge=0.0,   le=50.0,  description="Solution temperature (°C)")
    humidity:      float    = Field(..., ge=0.0,   le=100.0, description="Ambient humidity (%)")
    light_lux:     float    = Field(..., ge=0.0,              description="Light intensity (Lux)")
    growth_stage:  GrowthStage = GrowthStage.VEGETATIVE
    day_in_cycle:  int      = Field(..., ge=0,   le=365)
    hour_of_day:   int      = Field(..., ge=0,   le=23)
    nutrient_dose: float    = Field(0.0, ge=0.0,             description="Last nutrient dose applied (mL)")
    crop_type:     str      = "lettuce"


class ReadingOut(ReadingIn):
    id: str = Field(..., alias="_id")


# ─── LSTM Prediction ──────────────────────────────────────────────────────────

class LSTMPredictIn(BaseModel):
    """
    Request body for /predict/lstm.
    Either supply a window of recent readings directly,
    or pass `use_db=True` to fetch the latest N from MongoDB.
    """
    window:       list[dict[str, Any]] | None = None
    use_db:       bool                        = True
    lookback:     int                         = 24
    horizon:      int                         = Field(2, ge=1, le=8, description="Steps ahead (each = 15 min)")


class LSTMPredictOut(BaseModel):
    model:            str = "lstm"
    ph_forecast:      list[float]
    temp_forecast:    list[float]
    horizon_steps:    int
    minutes_per_step: int = 15
    ph_alert:         bool
    temp_alert:       bool
    created_at:       datetime = Field(default_factory=datetime.utcnow)


# ─── Nutrient Dosing (RL) ─────────────────────────────────────────────────────

class DosingIn(BaseModel):
    """State vector fed to the PPO RL agent."""
    ec_current:    float
    ph_current:    float
    growth_stage:  GrowthStage
    day_in_cycle:  int
    hour_of_day:   int
    crop_type:     str = "lettuce"


class DosingOut(BaseModel):
    model:         str = "ppo_rl"
    action:        DosingAction
    ec_target:     float
    ec_delta:      float
    confidence:    float
    reasoning:     str
    created_at:    datetime = Field(default_factory=datetime.utcnow)


# ─── Growth Stage (CNN) ───────────────────────────────────────────────────────

class GrowthOut(BaseModel):
    model:          str = "resnet50_cnn"
    growth_stage:   GrowthStage
    confidence:     float
    light_ppfd:     float   = Field(..., description="Recommended PPFD (µmol/m²/s)")
    photoperiod_h:  float   = Field(..., description="Recommended daily photoperiod (hours)")
    spectrum_note:  str
    created_at:     datetime = Field(default_factory=datetime.utcnow)


# ─── Pest Detection (YOLO) ────────────────────────────────────────────────────

class Detection(BaseModel):
    class_name:  str
    confidence:  float
    bbox:        list[float] = Field(..., description="[x1, y1, x2, y2] normalised 0-1")
    action:      str


class YOLOOut(BaseModel):
    model:       str = "yolov8"
    detections:  list[Detection]
    alert_level: AlertLevel
    summary:     str
    created_at:  datetime = Field(default_factory=datetime.utcnow)


# ─── Anomaly Detection (Autoencoder) ─────────────────────────────────────────

class AnomalyIn(BaseModel):
    """Single feature vector for anomaly scoring."""
    features: list[float] = Field(..., min_length=5)


class AnomalyOut(BaseModel):
    model:               str = "autoencoder"
    reconstruction_error: float
    threshold:           float
    is_anomaly:          bool
    alert_level:         AlertLevel
    created_at:          datetime = Field(default_factory=datetime.utcnow)


# ─── Alert ────────────────────────────────────────────────────────────────────

class AlertOut(BaseModel):
    id:        str
    source:    str
    level:     AlertLevel
    message:   str
    resolved:  bool
    created_at: datetime
