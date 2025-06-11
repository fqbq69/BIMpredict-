from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
import numpy as np
import json

app = FastAPI()

# Load models
ml_models = joblib.load("models/ml_models.joblib")  # Pre-trained ML models
dl_model = joblib.load("models/dl_model.joblib")  # Pre-trained DL model
preprocessor = joblib.load("models/preprocessor.joblib")  # Load preprocessing pipeline

@app.post("/predict/")
async def predict(data: dict):
    try:
        df = pd.DataFrame([data])  # Convert JSON input to DataFrame

        # Apply preprocessing (encoding, scaling, missing value handling)
        X_processed = preprocessor.transform(df)

        # Run predictions
        ml_preds = {name: model.predict(X_processed)[0] for name, model in ml_models.items()}
        dl_preds = dl_model.predict(X_processed).tolist()  # Convert DL outputs to list

        # Convert predictions to human-readable format
        response = {
            "ML_Predictions": ml_preds,
            "DL_Prediction": dl_preds
        }
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
