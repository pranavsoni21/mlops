import mlflow
import mlflow.sklearn
import pandas as pd
import os

from fastapi import FastAPI
from pydantic import BaseModel



app = FastAPI(title="Student Pass Prediction API")


# Load the model from MLflow
mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
)

model = mlflow.sklearn.load_model(
    "models:/StudentPassPredictor@champion"
)


class Student(BaseModel):
    hours_studied: float
    attendance: float
    previous_score: float
    sleep_hours: float


@app.get("/")
def root():
    return {"message": "Student Prediction API is running"}


@app.post("/predict")
def predict(student: Student):

    data = pd.DataFrame([{
        "hours_studied": student.hours_studied,
        "attendance": student.attendance,
        "previous_score": student.previous_score,
        "sleep_hours": student.sleep_hours
    }])

    prediction = model.predict(data)[0]

    return {
        "prediction": int(prediction),
        "result": "PASS" if prediction == 1 else "FAIL"
    }
