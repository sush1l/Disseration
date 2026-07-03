"""
models/ml.py — Model registry, lazy loading, inference, and SHAP explanation.

All models and SHAP explainers are loaded once at startup and cached.
Thread-safe for multi-worker deployments (read-only after init).
"""
from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

from api.config import (
    DEFAULT_MODEL,
    LABEL_ENCODER_PATH,
    MIN_CONFIDENCE,
    MODEL_PATHS,
    MODELS_NEEDING_SCALER,
    SCALER_PATH,
    SYMPTOMS_JSON_PATH,
    TOP_N_PREDICTIONS,
)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class ModelRegistry:
    """Loads and caches all ML artifacts on first access."""

    def __init__(self) -> None:
        self._models:         dict = {}
        self._explainers:     dict = {}          # model_key → shap.Explainer
        self._label_encoder   = None
        self._scaler          = None
        self._symptoms_meta:  dict = {}
        self._id_to_index:    dict[str, int] = {}
        self._index_to_id:    dict[int, str] = {}   # reverse for SHAP
        self._id_to_label:    dict[str, str] = {}   # symptom_id → human label
        self._n_features:     int = 0
        self._loaded:         bool = False

    # ── Public bootstrap ──────────────────────────────────────────────────────

    def load_all(self) -> None:
        """Called once at FastAPI startup."""
        self._load_symptoms()
        self._load_label_encoder()
        self._load_scaler()
        self._load_models()
        self._load_explainers()
        self._loaded = True

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(
        self,
        symptom_ids: list[str],
        model_key:   str = DEFAULT_MODEL,
        top_n:       int = TOP_N_PREDICTIONS,
    ) -> dict:
        if not self._loaded:
            raise RuntimeError("Registry not initialised — call load_all() first.")

        model_key = self._resolve_model_key(model_key)
        model     = self._models[model_key]

        matched, unrecognised, X = self._encode(symptom_ids)

        needs_scale = model_key in MODELS_NEEDING_SCALER or (
            model_key == "best_model" and DEFAULT_MODEL in MODELS_NEEDING_SCALER
        )
        if needs_scale and self._scaler is not None:
            X = self._scaler.transform(X).astype(np.float32)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_pred    = int(model.predict(X)[0])
            has_proba = hasattr(model, "predict_proba")
            if has_proba:
                probas = model.predict_proba(X)[0]
            else:
                scores = model.decision_function(X)[0]
                e      = np.exp(scores - scores.max())
                probas = e / e.sum()

        top_indices = np.argsort(probas)[::-1][:top_n]
        predictions = []
        for rank, idx in enumerate(top_indices, 1):
            conf = float(probas[idx])
            if conf < MIN_CONFIDENCE:
                break
            predictions.append({
                "rank":             rank,
                "disease":          self._label_encoder.classes_[idx],
                "confidence":       round(conf, 6),
                "confidence_pct":   f"{conf * 100:.2f}%",
            })

        top_class_idx  = int(top_indices[0]) if len(top_indices) else y_pred
        top_disease    = self._label_encoder.classes_[top_class_idx]
        top_confidence = round(float(probas[top_class_idx]), 6)

        return {
            "predictions":            predictions,
            "top_disease":            top_disease,
            "top_confidence":         top_confidence,
            "top_confidence_pct":     f"{top_confidence * 100:.2f}%",
            "model_used":             model_key,
            "symptoms_matched":       matched,
            "symptoms_unrecognised":  unrecognised,
            "symptom_count_matched":  len(matched),
            # passed to explain() by the route
            "_X":                     X,
            "_predicted_class_idx":   top_class_idx,
        }

    # ── SHAP explanation ──────────────────────────────────────────────────────

    def explain(
        self,
        X:                  np.ndarray,
        model_key:          str,
        predicted_class_idx: int,
        top_n:              int = 10,
    ) -> Optional[dict]:
        """
        Return the top SHAP factors for the predicted class.

        SHAP value interpretation
        ─────────────────────────
        A positive value means the symptom *pushed* the model toward this
        disease (evidence FOR).  A negative value means the symptom *pushed*
        the model away from this disease (evidence AGAINST).

        The baseline (expected_value) is the model's average log-odds / output
        before seeing any symptoms.  Each symptom's SHAP value is the shift it
        causes from that baseline, so the sum of all SHAP values + baseline
        equals the model's raw output for this prediction.
        """
        if not SHAP_AVAILABLE or model_key not in self._explainers:
            return None

        explainer = self._explainers[model_key]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = explainer.shap_values(X)
            class_shap = self._extract_class_shap(raw, predicted_class_idx)
        except Exception as e:
            print(f"  ⚠  SHAP inference failed for {model_key}: {e}")
            return None

        # Resolve base value (expected model output before seeing symptoms)
        try:
            ev = explainer.expected_value
            if isinstance(ev, (list, np.ndarray)):
                base_value = float(ev[min(predicted_class_idx, len(ev) - 1)])
            else:
                base_value = float(ev)
        except Exception:
            base_value = 0.0

        # Build per-symptom factor list
        factors = []
        for feat_idx, shap_val in enumerate(class_shap):
            sid = self._index_to_id.get(feat_idx)
            if sid is None:
                continue
            factors.append({
                "symptom_id":     sid,
                "symptom_label":  self._id_to_label.get(sid, sid.replace("_", " ").title()),
                "shap_value":     round(float(shap_val), 6),
                "feature_value":  float(X[0, feat_idx]),   # 1.0 = reported, 0.0 = absent
            })

        # Sort by absolute SHAP value descending
        factors.sort(key=lambda f: abs(f["shap_value"]), reverse=True)

        positive = [f for f in factors if f["shap_value"] > 0][:top_n]
        negative = [f for f in factors if f["shap_value"] < 0][:top_n]

        return {
            "base_value":          base_value,
            "predicted_class_idx": predicted_class_idx,
            "top_supporting":      positive,   # push prediction UP toward this disease
            "top_contradicting":   negative,   # push prediction DOWN away from this disease
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _extract_class_shap(self, shap_values, class_idx: int) -> np.ndarray:
        """Normalise SHAP output across model types to 1-D array (n_features,)."""
        if isinstance(shap_values, list):
            # Multi-class sklearn / tree: list[n_classes] of (n_samples, n_features)
            idx = min(class_idx, len(shap_values) - 1)
            arr = np.array(shap_values[idx])
            return arr[0] if arr.ndim == 2 else arr
        arr = np.array(shap_values)
        if arr.ndim == 3:
            # (n_samples, n_features, n_classes) — newer shap API
            return arr[0, :, min(class_idx, arr.shape[2] - 1)]
        if arr.ndim == 2:
            return arr[0]
        return arr

    def _encode(self, symptom_ids: list[str]) -> tuple[list, list, np.ndarray]:
        arr          = np.zeros((1, self._n_features), dtype=np.float32)
        matched      = []
        unrecognised = []
        for sid in symptom_ids:
            sid_clean = sid.strip().lower().replace(" ", "_").replace("-", "_")
            if sid_clean in self._id_to_index:
                arr[0, self._id_to_index[sid_clean]] = 1.0
                matched.append(sid_clean)
            else:
                unrecognised.append(sid)
        return matched, unrecognised, arr

    def _resolve_model_key(self, key: str) -> str:
        if key not in self._models:
            raise ValueError(f"Unknown model '{key}'. Available: {list(self._models.keys())}")
        return key

    def get_model_info(self) -> dict:
        return {
            "available_models": list(self._models.keys()),
            "default_model":    DEFAULT_MODEL,
            "n_features":       self._n_features,
            "n_diseases":       len(self._label_encoder.classes_) if self._label_encoder else 0,
            "diseases":         list(self._label_encoder.classes_) if self._label_encoder else [],
            "shap_available":   SHAP_AVAILABLE,
            "shap_models":      list(self._explainers.keys()),
            "performance": {
                "logistic_regression": {"accuracy": 0.8685, "f1": 0.8435, "auc": 0.9998},
                "random_forest":       {"accuracy": 0.5460, "f1": 0.6082, "auc": 0.9691},
                "decision_tree":       {"accuracy": 0.0301, "f1": 0.0971, "auc": None},
                "lightgbm":            {"accuracy": 0.0062, "f1": 0.0008, "auc": None},
            },
        }

    def get_symptoms(self) -> dict:
        return self._symptoms_meta

    # ── Loaders ───────────────────────────────────────────────────────────────

    def _load_symptoms(self) -> None:
        path = Path(SYMPTOMS_JSON_PATH)
        if not path.exists():
            raise FileNotFoundError(f"symptoms.json not found: {path}")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self._symptoms_meta = data
        self._id_to_index   = data["id_to_index"]
        self._n_features    = data["meta"]["total"]
        # Build reverse mappings for SHAP
        self._index_to_id   = {v: k for k, v in self._id_to_index.items()}
        self._id_to_label   = {s["id"]: s["label"] for s in data["flat"]}
        print(f"  ✓ symptoms.json — {self._n_features} symptoms, {len(data['categories'])} categories")

    def _load_label_encoder(self) -> None:
        path = Path(LABEL_ENCODER_PATH)
        if not path.exists():
            raise FileNotFoundError(f"label_encoder.pkl not found: {path}")
        with open(path, "rb") as f:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._label_encoder = pickle.load(f)
        print(f"  ✓ label_encoder — {len(self._label_encoder.classes_)} diseases")

    def _load_scaler(self) -> None:
        path = Path(SCALER_PATH)
        if not path.exists():
            print("  ⚠  scaler.pkl not found — models requiring scaling will skip it")
            return
        with open(path, "rb") as f:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._scaler = pickle.load(f)
        print("  ✓ scaler loaded")

    def _load_models(self) -> None:
        for key, path in MODEL_PATHS.items():
            p = Path(path)
            if not p.exists():
                print(f"  ⚠  {key}.pkl not found — skipping")
                continue
            size_mb = p.stat().st_size / 1024 / 1024
            print(f"  Loading {key} ({size_mb:.1f} MB)...", end=" ", flush=True)
            with open(p, "rb") as f:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self._models[key] = pickle.load(f)
            print("✓")
        if not self._models:
            raise RuntimeError("No models loaded. Check ARTIFACT_DIR in config.py.")

    def _load_explainers(self) -> None:
        if not SHAP_AVAILABLE:
            print("  ⚠  shap not installed — SHAP explanations disabled")
            return
        background = np.zeros((1, self._n_features), dtype=np.float32)
        print("  Building SHAP explainers...")
        for key, model in self._models.items():
            try:
                self._explainers[key] = self._make_explainer(model, background)
                print(f"  ✓ SHAP explainer — {key}")
            except Exception as e:
                print(f"  ⚠  SHAP explainer failed for {key}: {e}")

    def _make_explainer(self, model, background: np.ndarray):
        """Choose the fastest SHAP explainer for the given model type."""
        name = type(model).__name__
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if any(x in name for x in ("Logistic", "Linear", "Ridge", "SGD")):
                return shap.LinearExplainer(model, background)
            if any(x in name for x in ("Forest", "Tree", "Boosting", "LGBM", "Booster", "XGB")):
                return shap.TreeExplainer(model)
            # Generic fallback — may be slow for large models
            return shap.Explainer(model, background)


# Module-level singleton — imported by routes
registry = ModelRegistry()
