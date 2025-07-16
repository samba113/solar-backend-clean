# app.py
import joblib
import os
import gdown
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

model_path = "solar_power_model.joblib"

# Download model if not present
if not os.path.exists(model_path):
    url = "https://drive.google.com/uc?id=112VYQsoPWR8Wj3cT9IXN6euzrSqcayx0"  # your full model
    gdown.download(url, model_path, quiet=False)

model = joblib.load(model_path)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict_power(request: Request):
    data = await request.json()
    temp = float(data.get("temperature"))
    humidity = float(data.get("humidity"))
    pressure = float(data.get("pressure"))
    wind = float(data.get("windspeed"))

    prediction = model.predict([[temp, humidity, pressure, wind]])
    return {"prediction": round(prediction[0], 2)}
