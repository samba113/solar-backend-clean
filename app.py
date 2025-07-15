from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import gdown
import os
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "solar_power_model_compressed_v2.joblib"
DRIVE_FILE_ID = "112VYQsoPWR8Wj3cT9IXN6euzrSqcayx0"

if not os.path.exists(MODEL_PATH):
    print("📥 Downloading compressed model from Google Drive...")
    url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

model = joblib.load(MODEL_PATH)

class Query(BaseModel):
    message: str

def extract_features(text: str):
    pattern = r"(\d+(?:\.\d+)?)\D+(\d+(?:\.\d+)?)\D+(\d+(?:\.\d+)?)\D+(\d+(?:\.\d+)?)"
    m = re.search(pattern, text)
    if not m:
        raise ValueError("Could not extract all 4 inputs (temp, humidity, pressure, wind)")
    return list(map(float, m.groups()))

@app.post("/predict")
def predict(data: Query):
    try:
        features = extract_features(data.message)
        prediction = model.predict([features])[0]
        return {"prediction": f"⚡ Predicted solar power output: {prediction:.2f} W/m²"}
    except Exception as e:
        return {"prediction": f"❌ Error: {str(e)}"}

@app.get("/")
def root():
    return {"message": "✅ Backend is running. Use POST /predict"}
