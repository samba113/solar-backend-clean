from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import joblib
import gdown
import os
import re
from pydantic import BaseModel

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Google Drive download config
MODEL_PATH = "solar_power_model_compressed.joblib"
DRIVE_FILE_ID = "1YTbH3EvF_O0z5ncilUSHtyhSkJdit5Rn"

# Download the model if not already present
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

# Load model
model = joblib.load(MODEL_PATH)

# Request model
class Query(BaseModel):
    query: str

# Input extraction logic
def extract_features(query):
    query = query.lower()
    temp = re.search(r"(\d+\.?\d*)\s*(°?c|celsius|temp|temperature)", query)
    hum = re.search(r"(\d+\.?\d*)\s*%?\s*(humidity)", query)
    pres = re.search(r"(\d+\.?\d*)\s*(pressure|hpa)", query)
    wind = re.search(r"(\d+\.?\d*)\s*(m/s|wind)", query)

    if not (temp and hum and pres and wind):
        return None

    try:
        temp_val = float(temp.group(1))
        hum_val = float(hum.group(1))
        pres_val = float(pres.group(1))
        wind_val = float(wind.group(1))

        # Optional: sanity range checks
        if not (0 <= temp_val <= 60 and 0 <= hum_val <= 100 and 950 <= pres_val <= 1050 and 0 <= wind_val <= 30):
            return "⚠️ Input values are out of expected range."

        return [[temp_val, hum_val, pres_val, wind_val]]
    except:
        return None

# Prediction endpoint
@app.post("/predict")
async def predict(data: Query):
    query_text = data.query
    features = extract_features(query_text)

    if features is None:
        return {"error": "❌ Could not extract all 4 inputs (Temperature, Humidity, Pressure, Wind Speed)"}
    if isinstance(features, str):
        return {"error": features}

    prediction = model.predict(features)[0]
    return {"prediction": round(prediction, 2)}
