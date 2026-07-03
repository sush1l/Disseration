"""
routes/predict.py — Disease prediction endpoints.

POST /predict          → single prediction + SHAP explanation
POST /predict/batch    → multiple patients (no SHAP, for speed)
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from api.config import DEFAULT_MODEL, MODEL_PATHS, TOP_N_PREDICTIONS
from api.models.ml import registry

router = APIRouter(prefix="/predict", tags=["Prediction"])


# ─── Request / Response schemas ───────────────────────────────────────────────

class PredictRequest(BaseModel):
    symptoms: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "List of symptom IDs (snake_case, e.g. 'fever', 'cough'). "
            "Use GET /symptoms to browse valid IDs."
        ),
        examples=[["cough", "fever", "shortness_of_breath", "chest_tightness"]],
    )
    model: Optional[str] = Field(
        default=DEFAULT_MODEL,
        description=f"Model key. Default: '{DEFAULT_MODEL}'. Available: {list(MODEL_PATHS.keys())}",
    )
    top_n: int = Field(default=TOP_N_PREDICTIONS, ge=1, le=20)

    @field_validator("symptoms")
    @classmethod
    def check_symptoms(cls, v: list[str]) -> list[str]:
        if len(v) == 0:
            raise ValueError("At least one symptom must be provided.")
        if len(v) > 377:
            raise ValueError("Cannot provide more than 377 symptoms.")
        return [s.strip().lower() for s in v if s.strip()]


class PredictionItem(BaseModel):
    rank:           int
    disease:        str
    confidence:     float
    confidence_pct: str


# ── SHAP schemas ──────────────────────────────────────────────────────────────

class ShapFactor(BaseModel):
    symptom_id:    str
    symptom_label: str
    shap_value:    float
    feature_value: float   # 1.0 = symptom was reported, 0.0 = absent


class ShapExplanation(BaseModel):
    """
    SHAP (SHapley Additive exPlanations) breakdown for the top predicted disease.

    How to read this:
    - base_value: the model's average prediction across the training set (log-odds
      or probability, depending on model type). Think of it as "what the model
      predicts before seeing any symptoms."
    - top_supporting: symptoms that *increased* the probability of this disease
      (positive SHAP values). The higher the value, the stronger the push.
    - top_contradicting: symptoms that *decreased* the probability of this disease
      (negative SHAP values). Their absence or presence pulls the model away.
    - feature_value 1.0 means the patient reported this symptom; 0.0 means absent.
      An absent symptom can still have a high |SHAP| if the model expected it to
      be present for this disease.
    """
    base_value:           float
    predicted_class_idx:  int
    top_supporting:       list[ShapFactor]
    top_contradicting:    list[ShapFactor]


class PredictResponse(BaseModel):
    success:               bool
    top_disease:           str
    top_confidence:        float
    top_confidence_pct:    str
    predictions:           list[PredictionItem]
    model_used:            str
    symptoms_matched:      list[str]
    symptoms_unrecognised: list[str]
    symptom_count_matched: int
    shap_explanation:      Optional[ShapExplanation] = None


# ── Batch schemas (no SHAP — too slow for bulk) ───────────────────────────────

class BatchPatient(BaseModel):
    patient_id: str = Field(..., description="Your identifier for this patient/case.")
    symptoms:   list[str] = Field(..., min_length=1)

    @field_validator("symptoms")
    @classmethod
    def clean(cls, v: list[str]) -> list[str]:
        return [s.strip().lower() for s in v if s.strip()]


class BatchRequest(BaseModel):
    patients: list[BatchPatient] = Field(..., min_length=1, max_length=50)
    model:    Optional[str] = Field(default=DEFAULT_MODEL)
    top_n:    int = Field(default=3, ge=1, le=10)


class BatchResultItem(BaseModel):
    patient_id:  str
    top_disease: str
    confidence:  float
    predictions: list[PredictionItem]
    error:       Optional[str] = None


class BatchResponse(BaseModel):
    success:    bool
    results:    list[BatchResultItem]
    model_used: str


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post(
    "",
    response_model=PredictResponse,
    summary="Predict disease from symptoms (includes SHAP explanation)",
)
async def predict_disease(body: PredictRequest) -> PredictResponse:
    model_key = body.model or DEFAULT_MODEL

    try:
        result = registry.predict(
            symptom_ids=body.symptoms,
            model_key=model_key,
            top_n=body.top_n,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    if not result["symptoms_matched"]:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "None of the provided symptoms were recognised.",
                "unrecognised": result["symptoms_unrecognised"],
                "hint": "Use GET /symptoms to browse valid symptom IDs.",
            },
        )

    # SHAP explanation for the top predicted disease
    shap_data = registry.explain(
        X=result.pop("_X"),
        model_key=model_key,
        predicted_class_idx=result.pop("_predicted_class_idx"),
    )

    shap_explanation = ShapExplanation(**shap_data) if shap_data else None

    return PredictResponse(
        success=True,
        shap_explanation=shap_explanation,
        **result,
    )


@router.post(
    "/batch",
    response_model=BatchResponse,
    summary="Batch predict for multiple patients (no SHAP)",
)
async def predict_batch(body: BatchRequest) -> BatchResponse:
    model_key = body.model or DEFAULT_MODEL
    results   = []

    for patient in body.patients:
        try:
            result = registry.predict(
                symptom_ids=patient.symptoms,
                model_key=model_key,
                top_n=body.top_n,
            )
            result.pop("_X", None)
            result.pop("_predicted_class_idx", None)
            results.append(BatchResultItem(
                patient_id=patient.patient_id,
                top_disease=result["top_disease"],
                confidence=result["top_confidence"],
                predictions=[PredictionItem(**p) for p in result["predictions"]],
            ))
        except Exception as e:
            results.append(BatchResultItem(
                patient_id=patient.patient_id,
                top_disease="",
                confidence=0.0,
                predictions=[],
                error=str(e),
            ))

    return BatchResponse(success=True, results=results, model_used=model_key)
