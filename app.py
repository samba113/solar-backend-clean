from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os
import gdown
import re
import numpy as np

app = FastAPI()

# CORS for frontend (React/Streamlit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # OR use your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Download and load model
MODEL_URL = "https://drive.google.com/uc?id=1YTbH3EvF_O0z5ncilUSHtyhSkJdit5Rn"
MODEL_PATH = "solar_power_model.joblib"

if not os.path.exists(MODEL_PATH):
    print("📥 Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

print("✅ Loading model...")
model = joblib.load(MODEL_PATH)

# Input model for structured requests
class WeatherInput(BaseModel):
    temperature: float
    humidity: float
    pressure: float
    windspeed: float

# NLP parser
def extract_weather_data(text):
    numbers = list(map(float, re.findall(r"[-+]?\d*\.\d+|\d+", text)))
    if len(numbers) >= 4:
        return {
            "temperature": numbers[0],
            "humidity": numbers[1],
            "pressure": numbers[2],
            "windspeed": numbers[3]
        }
    return None

@app.get("/")
def home():
    return {
        "message": "🤖 Solar AI Chatbot: Use /predict with structured or natural input"
    }

@app.post("/predict")
async def predict(request: Request):
    try:
        body = await request.json()

        # Handle NLP input from chatbot
        if isinstance(body, dict) and "message" in body:
            user_text = body["message"]
            extracted = extract_weather_data(user_text)
            if not extracted:
                return {"error": "❌ Could not extract all weather values. Use: Temp 30, humidity 50, pressure 1005, wind 2.5"}
            data = extracted
        else:
            data = body

        input_data = [[
            float(data["temperature"]),
            float(data["humidity"]),
            float(data["pressure"]),
            float(data["windspeed"])
        ]]
        prediction = model.predict(input_data)[0]
        return {
            "prediction": float(prediction)
        }
    except Exception as e:
        return {"error": f"❌ Error: {str(e)}"}
