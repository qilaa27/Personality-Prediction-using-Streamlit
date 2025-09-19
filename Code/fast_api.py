import uvicorn
from fastapi import FastAPI
from personality import PersonalityBase
import pandas as pd
import joblib

# Load trained pipeline model
model = joblib.load("trained_model.pkl")

# Define label mapping
label_map = {0: "Extrovert", 1: "Introvert"}

# Initialize app
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Personality Prediction API is running"}

@app.post("/predict")
def predict(data: PersonalityBase):
    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Predict
    prediction = int(model.predict(input_df)[0])
    label = label_map.get(prediction, "Unknown")

    return {
        "prediction": prediction,
        "label": label
    }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=7860)

    