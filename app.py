from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
import joblib, gdown, os, re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

MODEL_PATH = "solar_power_model_compressed_v2.joblib"
DRIVE_FILE_ID = "112VYQsoPWR8Wj3cT9IXN6euzrSqcayx0"

if not os.path.exists(MODEL_PATH):
    gdown.download(f"https://drive.google.com/uc?id={DRIVE_FILE_ID}", MODEL_PATH, quiet=False)

model = joblib.load(MODEL_PATH)

class Query(BaseModel):
    message: str
    model_config = ConfigDict(from_attributes=True)

def extract_features(text: str):
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    if len(nums) < 4:
        raise ValueError("Need 4 numbers: Temp, Humidity, Pressure, Wind")
    return list(map(float, nums[:4]))

@app.post("/predict")
def predict(data: Query):
    try:
        features = extract_features(data.message)
        pred = model.predict([features])[0]
        return {"prediction": f"⚡ Predicted solar power output: {pred:.2f} W/m²"}
    except Exception as e:
        return {"prediction": f"❌ Error: {str(e)}"}

@app.get("/")
def root():
    return {"message": "✅ Solar backend running"}
