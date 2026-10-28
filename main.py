
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Bengaluru House Price Prediction API",
    description="Predict Bengaluru house prices using machine learning",
    version="1.0.0",
)


class HouseInput(BaseModel):
    location: str = Field(min_length=1)
    total_sqft: float = Field(gt=0)
    bhk: float = Field(gt=0)
    bath: float = Field(gt=0)


@app.get("/")
def home():
    return {
        "message": "Bengaluru House Price Prediction API",
        "status": "running",
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
def predict_price(house: HouseInput):
    location = house.location.strip()

    if not location:
        raise HTTPException(
            status_code=422,
            detail="Location cannot be empty",
        )

    input_data = pd.DataFrame([{
        "location": location,
        "total_sqft": house.total_sqft,
        "bhk": house.bhk,
        "bath": house.bath,
    }])

    prediction = model.predict(input_data)[0]

    return {
        "location": location,
        "predicted_price_lakhs": round(float(prediction), 2),
        "currency": "INR",
        "unit": "lakhs",
    }
