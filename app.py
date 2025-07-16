# app.py
import joblib
import os
import gdown
import re
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

model_path = "solar_power_model.joblib"

# Download model if not present
if not os.path.exists(model_path):
    print("📥 Downloading model from Google Drive...")
    url = "https://drive.google.com/uc?id=112VYQsoPWR8Wj3cT9IXN6euzrSqcayx0"
    gdown.download(url, model_path, quiet=False)

model = joblib.load(model_path)
print("✅ Model loaded successfully")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🧠 NLP function to extract values
def extract_weather_values(text):
    numbers = re.findall(r"\d+\.?\d*", text)
    if len(numbers) >= 4:
        temp, humidity, pressure, windspeed = map(float, numbers[:4])
        return temp, humidity, pressure, windspeed
    else:
        return None

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    message = data.get("message")

    result = extract_weather_values(message)
    if result:
        temp, humidity, pressure, windspeed = result
        prediction = model.predict([[temp, humidity, pressure, windspeed]])
        return {"prediction": round(float(prediction[0]), 2)}
    else:
        return {"error": "❌ Could not extract all weather values. Please follow the format like:\n- Temp 30, humidity 50, pressure 1005, wind 2.5"}
