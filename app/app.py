import mlflow
import mlflow.sklearn
import pandas as pd
import os
import time
import logging
import json

from mlflow.tracking import MlflowClient
from datetime import datetime, timezone
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, make_asgi_app


app = FastAPI(title="Student Pass Prediction API")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("prediction-api")


# Total successful predictions, grouped by result
prediction_requests = Counter(
    "prediction_requests_total",
    "Total successful prediction requests",
    ["result"]
)

# HTTP requests that return an error
prediction_errors = Counter(
    "prediction_request_errors_total",
    "Total HTTP requests returning an error",
    ["status_code"]
)

# Time spent processing HTTP requests
prediction_duration = Histogram(
    "prediction_request_duration_seconds",
    "HTTP request processing duration",
    ["method", "endpoint"]
)

# Expose metrics for Prometheus scraping
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

@app.middleware("http")
async def monitor_requests(request, call_next):
    start_time = time.perf_counter()

    response = await call_next(request)

    duration = time.perf_counter() - start_time

    endpoint = request.url.path

    prediction_duration.labels(
        method=request.method,
        endpoint=endpoint
    ).observe(duration)

    if response.status_code >= 400:
        prediction_errors.labels(
            status_code=str(response.status_code)
        ).inc()

    return response


# Load the model from MLflow
mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
)

MODEL_NAME = "StudentPassPredictor"
MODEL_ALIAS = "champion"

client = MlflowClient()

model_version = client.get_model_version_by_alias(
    MODEL_NAME,
    MODEL_ALIAS
)

MODEL_VERSION = model_version.version

model = mlflow.sklearn.load_model(
    f"models:/{MODEL_NAME}/{MODEL_VERSION}"
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

    prediction = int(model.predict(data)[0])
    result = "PASS" if prediction == 1 else "FAIL"

    prediction_requests.labels(result=result).inc()

    log_entry = {
        "event": "prediction",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
        "model_version": MODEL_VERSION,
        "features": data.iloc[0].to_dict(),
        "prediction": prediction,
        "result": result
    }

    logger.info(json.dumps(log_entry))

    return {
        "prediction": int(prediction),
        "result": "PASS" if prediction == 1 else "FAIL"
    }
