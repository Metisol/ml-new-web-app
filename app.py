from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import joblib
import pandas as pd
from pydantic import BaseModel

# ✅ Load the trained model
try:
    model = joblib.load("model.pkl")
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    raise RuntimeError("Failed to load the model")

# ✅ Initialize FastAPI
app = FastAPI()

# ✅ Serve static files (Frontend UI)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ✅ Redirect root URL ("/") to `index.html`
@app.get("/")
def read_root():
    return FileResponse("static/index.html")

class PredictionRequest(BaseModel):
    AGE: int
    TOTAL_VOTES: int
    GENERAL_VOTES: int
    POSTAL_VOTES: int
    TOTAL_ELECTORS: int
    CRIMINAL_CASES: int
    ASSETS: float
    LIABILITIES: float
    EDUCATION: int
    CATEGORY: int
    GENDER: int
    PARTY: int

@app.post("/predict")
def predict(request: PredictionRequest):
    input_data = pd.DataFrame([request.dict()])
    prediction = model.predict(input_data)[0]
    return {"Prediction": "🎉 Won" if prediction == 1 else "❌ Lost"}

@app.get("/health")
def health_check():
    return {"status": "API is running!"}
