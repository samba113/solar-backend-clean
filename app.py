import os
import joblib
import gdown
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

model_path = "solar_power_model.joblib"

# ✅ Download model if not already present
if not os.path.exists(model_path):
    print("📥 Downloading model from Google Drive...")
    gdown.download(
        "https://drive.google.com/uc?id=112VYQsoPWR8Wj3cT9IXN6euzrSqcayx0",
        model_path,
        quiet=False,
    )

# ✅ Load model
print("✅ Loading model...")
model = joblib.load(model_path)

# ✅ FastAPI setup
app = FastAPI()

# ✅ Allow all CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict_power(request: Request):
    data = await request.json()

    try:
        temp = float(data.get("temperature"))
        humidity = float(data.get("humidity"))
        pressure = float(data.get("pressure"))
        wind = float(data.get("windspeed"))

        prediction = model.predict([[temp, humidity, pressure, wind]])

        # ✅ Convert to native Python float for FastAPI
        result = float(round(prediction[0], 2))

        return {"prediction": result}

    except Exception as e:
        return {"error": f"❌ Failed to predict: {str(e)}"}
