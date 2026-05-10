import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import yaml

from ollama_insights import check_ollama_available, generate_model_comparison


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── Request / Response schemas ─────────────────────────────────────────────────

class EmployeeFeatures(BaseModel):
    Age: int
    BusinessTravel: int
    DailyRate: int
    Department: int
    DistanceFromHome: int
    Education: int
    EducationField: int
    EnvironmentSatisfaction: int
    Gender: int
    HourlyRate: int
    JobInvolvement: int
    JobLevel: int
    JobRole: int
    JobSatisfaction: int
    MaritalStatus: int
    MonthlyIncome: int
    MonthlyRate: int
    NumCompaniesWorked: int
    OverTime: int
    PercentSalaryHike: int
    PerformanceRating: int
    RelationshipSatisfaction: int
    StockOptionLevel: int
    TotalWorkingYears: int
    TrainingTimesLastYear: int
    WorkLifeBalance: int
    YearsAtCompany: int
    YearsInCurrentRole: int
    YearsSinceLastPromotion: int
    YearsWithCurrManager: int

class PredictionResponse(BaseModel):
    model_used: str
    attrition_probability: float
    attrition_prediction: str
    risk_level: str

class HealthResponse(BaseModel):
    status: str
    models_loaded: list[str]
    ollama_available: bool


# ── Application ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Employee Attrition Prediction API",
    description="Predicts employee attrition risk using trained ML classifiers.",
    version="1.0.0",
)

config = load_config()
models_dir = Path(config["training"]["output_dir"])

# Load artifacts at startup — avoids reloading on every request
_models = {}
_scaler = None

@app.on_event("startup")
def load_artifacts():
    """Load trained models and scaler from disk at server startup."""
    global _scaler, _models

    scaler_path = models_dir / "scaler.pkl"
    if not scaler_path.exists():
        return

    _scaler = joblib.load(scaler_path)

    for model_file in models_dir.glob("*.pkl"):
        if model_file.stem != "scaler":
            _models[model_file.stem] = joblib.load(model_file)

    print(f"[api] Loaded models: {list(_models.keys())}")


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    """Liveness probe — confirms API is running and reports loaded model state."""
    return HealthResponse(
        status="ok",
        models_loaded=list(_models.keys()),
        ollama_available=check_ollama_available(config["ollama"]["host"]),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(features: EmployeeFeatures, model_name: str = "random_forest"):
    """
    Predict attrition probability for a single employee.
    Accepts preprocessed (encoded) feature values matching the training schema.
    """
    if not _models:
        raise HTTPException(status_code=503, detail="No trained models loaded. Run training first.")

    if model_name not in _models:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model_name}'. Available: {list(_models.keys())}"
        )

    model = _models[model_name]
    input_array = np.array([list(features.model_dump().values())])
    input_scaled = _scaler.transform(input_array)

    probability = model.predict_proba(input_scaled)[0][1]
    prediction = "Yes" if probability >= 0.5 else "No"

    # Risk bucketing for human-readable output
    if probability >= 0.7:
        risk = "High"
    elif probability >= 0.4:
        risk = "Medium"
    else:
        risk = "Low"

    return PredictionResponse(
        model_used=model_name,
        attrition_probability=round(probability, 4),
        attrition_prediction=prediction,
        risk_level=risk,
    )


@app.get("/models")
def list_models():
    """List all available trained models."""
    return {"available_models": list(_models.keys())}