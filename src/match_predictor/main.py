from typing import Any, Dict, List

import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


class PredictRequest(BaseModel):
    data: List[Dict[str, Any]]


# Model info
model_name = "tennis_predictor_model"

# Load the model
clf = mlflow.xgboost.load_model(f"models:/{model_name}@champion")

# create FastAPI object
app = FastAPI()


# API operations
@app.get("/")
def health_check():
    return {"health_check": "OK"}


@app.get("/info")
def info():
    return {
        "name": "tennis_predictor",
        "description": "Predict the outcome of tennis matches.",
    }


@app.post("/predict")
async def predict(request: PredictRequest) -> dict:
    """
    Predict the outcome of tennis matches.

    request: JSON with key 'data', a list of dicts (one per match).
    return: Prediction results.
    """
    # Convert the input list of dicts to a DataFrame
    data_df = pd.DataFrame(request.data)
    # convert the object columns to float
    for col in data_df.select_dtypes(include=["object"]).columns:
        data_df[col] = pd.to_numeric(data_df[col], errors="coerce")
    # Make predictions
    predictions = clf.predict(data_df)

    return {"predictions": predictions.tolist()}
