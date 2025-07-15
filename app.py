from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import joblib
import os
import gdown
import re

app = FastAPI()

# Enable CORS for all domains (React frontend etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Compressed model path and URL (Google Drive)
MODEL_PATH = "solar_power_model_compressed_v2.joblib"
MODEL_URL = "https://drive.google.com/uc?id=112VYQsoPWR8Wj3cT9IXN6euzrSqcayx0"

# Download the model only if not already downloaded
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading compressed model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load the compressed model
model = joblib.load(MODEL_PATH)

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    message = data.get("message", "")

    # Extract temperature, humidity, pressure, wind speed
    pattern = r"(\d+(?:\.\d+)?)\D+(\d+(?:\.\d+)?)%\D+(\d+(?:\.\d+)?)\D+(\d+(?:\.\d+)?)"
    match = re.search(pattern, message)

    if match:
        temp = float(match.group(1))
        humidity = float(match.group(2))
        pressure = float(match.group(3))
        wind_speed = float(match.group(4))

        # Validate range
        if not (950 <= pressure <= 1050):
            return {"response": "⚠️ Pressure must be between 950 and 1050 hPa"}

        prediction = model.predict([[temp, humidity, pressure, wind_speed]])[0]
        return {"response": f"⚡ Predicted solar power output: {prediction:.2f} W/m²"}

    return {"response": "❌ Could not extract all required values (Temp, Humidity%, Pressure, Wind Speed)"}
