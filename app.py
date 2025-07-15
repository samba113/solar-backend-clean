from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import re
import os
import gdown

app = FastAPI()

# Allow frontend requests (from localhost or deployment)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔽 Google Drive model download
MODEL_PATH = "solar_power_model.joblib"
MODEL_URL = "https://drive.google.com/uc?id=1YTbH3EvF_O0z5ncilUSHtyhSkJdit5Rn"

if not os.path.exists(MODEL_PATH):
    print("📥 Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load model
model = joblib.load(MODEL_PATH)

# Request schema
class QueryInput(BaseModel):
    query: str

# Prediction endpoint
@app.post("/predict")
async def predict(data: QueryInput):
    query = data.query

    try:
        # Extract features using regex
        temp_match = re.search(r"(\d+(?:\.\d+)?)\s*°?C?", query)
        hum_match = re.search(r"(\d+(?:\.\d+)?)\s*%?\s*humidity", query)
        pres_match = re.search(r"(\d+(?:\.\d+)?)\s*h?pa|pressure", query)
        wind_match = re.search(r"(\d+(?:\.\d+)?)\s*(m/s|wind)", query)

        if not (temp_match and hum_match and pres_match and wind_match):
            return {"error": "❌ Could not extract all 4 inputs (Temperature, Humidity, Pressure, Wind Speed)"}

        temperature = float(temp_match.group(1))
        humidity = float(hum_match.group(1))
        pressure = float(pres_match.group(1))
        wind_speed = float(wind_match.group(1))

        # Optional: Add range checks
        if not (0 <= temperature <= 60):
            return {"error": "⚠️ Temperature out of range (0–60°C)"}
        if not (0 <= humidity <= 100):
            return {"error": "⚠️ Humidity out of range (0–100%)"}
        if not (950 <= pressure <= 1050):
            return {"error": "⚠️ Pressure out of range (950–1050 hPa)"}
        if not (0 <= wind_speed <= 15):
            return {"error": "⚠️ Wind speed out of range (0–15 m/s)"}

        # Predict
        prediction = model.predict([[temperature, humidity, pressure, wind_speed]])[0]
        return {"prediction": round(prediction, 2)}

    except Exception as e:
        return {"error": f"❌ Error extracting input features: {str(e)}"}
