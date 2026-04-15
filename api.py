from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from housing_prediction import DEFAULT_MODEL_OUTPUT_PATH, load_model_artifact, predict_from_artifact


MODEL_PATH = Path(DEFAULT_MODEL_OUTPUT_PATH)

app = FastAPI(title="Housing Prediction API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:18.188.142.26",
        "http://localhost:18.188.142.26:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

artifact: dict[str, Any] = {}


@app.on_event("startup")
def load_model() -> None:
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")

    artifact.clear()
    artifact.update(load_model_artifact(MODEL_PATH))
    print(f"Model loaded - type: {artifact['model_type']}")
    print(f"Regions: {artifact['regions']}")
    print(f"Features: {artifact['feature_columns']}")


class PredictionRequest(BaseModel):
    features: dict[str, float | str | None]


class PredictionResponse(BaseModel):
    predicted_median_sale_price: float
    formatted: str
    region_used: str
    prediction_mode: str


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model_loaded": bool(artifact)}


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    if not artifact:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_type": artifact["model_type"],
        "target_column": artifact["target_column"],
        "feature_columns": artifact["feature_columns"],
        "metrics": artifact["metrics"],
        "regions": artifact["regions"],
        "default_region": artifact["default_region"],
        "forecast_horizon": artifact["forecast_horizon"],
        "training_start_by_region": artifact["training_start_by_region"],
        "training_end_by_region": artifact["training_end_by_region"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    if not artifact:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        prediction = predict_from_artifact(artifact, request.features)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return PredictionResponse(**prediction)


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
