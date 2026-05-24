import os
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")
os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI")

app = FastAPI(title="API Rekomendasi Lari")

class RunnerInput(BaseModel):
    umur: int
    speed: float
    distance: float
    gender: str

# Mengambil model dengan alias champion persis seperti arahan modul
model_name = os.getenv("MODEL_NAME", "Model_Rekomendasi_Lari")
model_alias = os.getenv("MODEL_ALIAS", "champion")
model_uri = f"models:/{model_name}@{model_alias}"

try:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    print("Sedang mengambil model dari MLflow...")
    model = mlflow.pyfunc.load_model(model_uri)
    print("Model berhasil dimuat!")
except Exception as e:
    print(f"Error loading model: {e}")

@app.get("/")
def home():
    return {"message": "Selamat datang di API Rekomendasi Lari Berbasis AI!"}

@app.post("/predict")
def predict(data: RunnerInput):
    df = pd.DataFrame([{
        'Athlete age': data.umur,
        'Athlete average speed': data.speed,
        'Distance_num': data.distance,
        'Athlete gender': data.gender
    }])
    prediksi = model.predict(df)[0]
    return {
        "status": "success",
        "input": data.dict(),
        "prediksi_level": str(prediksi)
    }