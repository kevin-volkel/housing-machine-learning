import pickle
from pathlib import Path

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

MODEL_PATH = Path("redfin_median_sale_price_model.pkl")

app = FastAPI(title="Housing Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
artifact: dict = {}


@app.on_event("startup")
def load_model() -> None:
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        artifact.update(pickle.load(f))
    print(f"Model loaded — target: {artifact['target_column']}")
    print(f"Features: {artifact['feature_columns']}")


class PredictionRequest(BaseModel):
    features: dict[str, float | str]


class PredictionResponse(BaseModel):
    predicted_median_sale_price: float
    formatted: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": bool(artifact)}


@app.get("/model-info")
def model_info() -> dict:
    if not artifact:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "target_column": artifact["target_column"],
        "feature_columns": artifact["feature_columns"],
        "metrics": artifact["metrics"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    if not artifact:
        raise HTTPException(status_code=503, detail="Model not loaded")

    pipeline = artifact["pipeline"]
    feature_columns = artifact["feature_columns"]

    # Build input row aligned to trained feature columns
    row = {col: request.features.get(col, None) for col in feature_columns}
    X = pd.DataFrame([row])

    try:
        prediction = float(pipeline.predict(X)[0])
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {exc}") from exc

    return PredictionResponse(
        predicted_median_sale_price=round(prediction, 2),
        formatted=f"${prediction:,.0f}",
    )


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
