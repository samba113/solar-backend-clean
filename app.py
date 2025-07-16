# app.py
import os
import joblib
import gdown
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# === Define model path ===
model_path = "solar_power_model.joblib"

# === Download model if not present ===
if not os.path.exists(model_path):
    print("📥 Downloading model from Google Drive...")
    gdown.download(
        "https://drive.google.com/uc?id=112VYQsoPWR8Wj3cT9IXN6euzrSqcayx0",
        model_path,
        quiet=False
    )

# === Load the model ===
print("✅ Loading model...")
model = joblib.load(model_path)

# === FastAPI setup ===
app = FastAPI()

# === CORS middleware ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"] in dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Request schema ===
class WeatherData(BaseModel):
    temperature: float
    humidity: float
    pressure: float
    windspeed: float

# === Prediction endpoint ===
@app.post("/predict")
async def predict_power(data: WeatherData):
    try:
        prediction = model.predict([[data.temperature, data.humidity, data.pressure, data.windspeed]])
        return {"prediction": round(prediction[0], 2)}
    except Exception as e:
        return {"error": str(e)}
