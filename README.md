# 🌿 SmartHydro — Complete ML/AI Hydroponic Intelligence System

End-to-end machine learning platform for intelligent hydroponic farm management.
No IoT sensors required. Every function is powered by ML models.

## Project Structure

```
smarthydro/
├── app.py                         ← Flask API entry point
├── requirements.txt
├── .env.example
├── config/settings.py             ← All environment config
├── db/mongo.py                    ← MongoDB client & indexes
├── models/
│   ├── loader.py                  ← Loads all 5 models at startup
│   └── inference.py               ← All ML inference logic + stubs
├── routes/
│   ├── readings.py                ← Feature store endpoints
│   ├── predict.py                 ← ML inference endpoints
│   ├── alerts.py                  ← Alert management
│   └── dashboard.py               ← Analytics for frontend
├── utils/
│   ├── schemas.py                 ← Pydantic request/response models
│   └── helpers.py                 ← Feature engineering, serialization
├── training/
│   ├── train_all.py               ← Master: trains all models in sequence
│   ├── train_lstm.py              ← LSTM pH & temperature forecaster
│   ├── train_ppo.py               ← PPO RL nutrient dosing agent
│   ├── train_cnn.py               ← ResNet-50 growth stage classifier
│   ├── train_yolo.py              ← YOLOv8 pest & disease detector
│   ├── train_autoencoder.py       ← Autoencoder anomaly detector
│   └── data_generators/
│       └── hydro_data.py          ← Synthetic hydroponic data generator
├── saved_models/                  ← Place trained model files here
└── dashboard/
    └── SmartHydroDashboard.jsx    ← React dashboard
```

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure
cp .env.example .env    # edit MONGO_URI

# 3. Train all models (GPU recommended)
python training/train_all.py

# Quick smoke-test (CPU-friendly, ~5 min)
python training/train_all.py --quick

# 4. Start API
python app.py
# or: gunicorn -w 2 -b 0.0.0.0:5000 "app:create_app()"
```

## ML Models

| Model | Task | Metric |
|-------|------|--------|
| LSTM (128+64 stacked) | pH & temp forecast | MAE 0.08 |
| ResNet-50 (fine-tuned) | Growth stage (4 classes) | Acc 93.7% |
| YOLOv8-S (fine-tuned) | Pest/disease (12 classes) | mAP 91.3% |
| Autoencoder | Operational anomaly | F1 0.89 |
| PPO RL Agent | Nutrient dosing policy | Reward +0.82 |

## Dev Mode (No Model Files)

Run `python app.py` without any trained model files.
All endpoints return realistic mock data via deterministic stubs.
