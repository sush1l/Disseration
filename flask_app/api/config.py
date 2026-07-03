"""
config.py — Central configuration for the prediction API.
All paths are relative to this file's location (api/).
"""
from pathlib import Path

# ── Directories ──────────────────────────────────────────────────────────────
API_DIR       = Path(__file__).parent
ARTIFACT_DIR  = API_DIR.parent.parent / "model_artifacts"

# ── Model artifacts ───────────────────────────────────────────────────────────
MODEL_PATHS = {
    "logistic_regression": ARTIFACT_DIR / "logistic_regression.pkl",
    "random_forest":       ARTIFACT_DIR / "random_forest.pkl",
    "decision_tree":       ARTIFACT_DIR / "decision_tree.pkl",
    "lightgbm":            ARTIFACT_DIR / "lightgbm.pkl",
    "best_model":          ARTIFACT_DIR / "best_model.pkl",
}

LABEL_ENCODER_PATH  = ARTIFACT_DIR / "label_encoder.pkl"
SCALER_PATH         = ARTIFACT_DIR / "scaler.pkl"
SYMPTOMS_JSON_PATH  = API_DIR / "symptoms.json"

# ── Model to use for /predict (change if you retrain) ────────────────────────
DEFAULT_MODEL = "best_model"   # must be a key in MODEL_PATHS

# ── Models that require StandardScaler before predict() ──────────────────────
MODELS_NEEDING_SCALER = {"logistic_regression"}

# ── Inference settings ────────────────────────────────────────────────────────
TOP_N_PREDICTIONS   = 5        # how many top diseases to return
MIN_CONFIDENCE      = 0.01     # drop predictions below this threshold (1 %)

# ── API metadata ──────────────────────────────────────────────────────────────
API_TITLE       = "Multi-Disease Prediction API"
API_DESCRIPTION = """
Predicts diseases from a set of reported symptoms using ML models trained on
the Dhivyesh Diseases and Symptoms Dataset (Kaggle).

**Models available:** Logistic Regression · Random Forest · Decision Tree · LightGBM
**Best model:** Logistic Regression (Accuracy 86.9 %, F1 84.4 %, AUC 0.9998)
"""
API_VERSION     = "1.0.0"
