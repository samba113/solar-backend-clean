from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import gdown
import os
import re

# ── FastAPI app ─────────────────────────────────────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # allow any frontend
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Download + load compressed model from Google Drive ──────
MODEL_PATH = "solar_power_model_compressed_v2.joblib"
DRIVE_FILE_ID = "112VYQsoPWR8Wj3cT9IXN6euzrSqcayx0"    # <- Your file ID

if not os.path.exists(MODEL_PATH):
    print("📥 Downloading compressed model from Google Drive...")
    url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

model = joblib.load(MODEL_PATH)

# ── Request schema ──────────────────────────────────────────
class Query(BaseModel):
    message: str

# ── Helper: extract 4 numbers (T, H, P, W) from text ────────
def extract_features(text: str):
    pattern = r"(\d+(?:\.\d+)?)\D+(\d+(?:\.\d+)?)\D+(\d+(?:\.\d+)?)\D+(\d+(?:\.\d+)?)"
    m = re.search(pattern, text)
    if not m:
        raise ValueError("Could not extract all 4 inputs")
    temp, hum, pres, wind = map(float, m.groups())
    return [temp, hum, pres, wind]

# ── Prediction endpoint ─────────────────────────────────────
@app.post("/predict")
def predict(data: Query):
    try:
        feats = extract_features(data.message)
        pred  = model.predict([feats])[0]
        return {"prediction": f"⚡ Predicted solar power output: {pred:.2f} W/m²"}
    except Exception as e:
        return {"prediction": f"❌ Error: {str(e)}"}

# ── Root route for Render health check ─────────────────────
@app.get("/")
def root():
    return {"message": "✅ Solar backend running"}
