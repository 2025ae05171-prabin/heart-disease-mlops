
# Import the libraries required for the API
import logging
import time
from datetime import datetime, timezone

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field


# --------------------------------------------------
# 1. Configure logging
# --------------------------------------------------

# Log messages to the container console with timestamps and severity.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("heart_disease_api")


# --------------------------------------------------
# 2. Create the FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Heart Disease Prediction API",
    description="MLOps assignment prediction service with logging",
    version="1.1.0",
)


# --------------------------------------------------
# 3. Load the trained model
# --------------------------------------------------

# The Dockerfile copies this model into the /app directory.
model = joblib.load("heart_disease_model.joblib")

logger.info("Heart disease model loaded successfully")


# --------------------------------------------------
# 4. Define the expected patient input
# --------------------------------------------------

# These 13 fields must match the features used during training.
class PatientData(BaseModel):
    age: float = Field(ge=0, le=120)
    sex: float
    cp: float
    trestbps: float = Field(ge=0)
    chol: float = Field(ge=0)
    fbs: float
    restecg: float
    thalach: float = Field(ge=0)
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float


# --------------------------------------------------
# 5. Store basic in-memory monitoring counters
# --------------------------------------------------

# These counters reset when the container restarts.
# For a larger production service, use persistent monitoring.
metrics = {
    "total_requests": 0,
    "successful_predictions": 0,
    "failed_requests": 0,
    "prediction_class_0": 0,
    "prediction_class_1": 0,
}


# --------------------------------------------------
# 6. Log every HTTP request
# --------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Record when the request started.
    start_time = time.perf_counter()

    # Count every incoming request, including health and metrics checks.
    metrics["total_requests"] += 1

    try:
        # Pass the request to the appropriate API endpoint.
        response = await call_next(request)

        # Count HTTP errors as failed requests.
        if response.status_code >= 400:
            metrics["failed_requests"] += 1

        # Calculate request-processing time in milliseconds.
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Record the method, path, status, and processing time.
        logger.info(
            "method=%s path=%s status=%s duration_ms=%.2f",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )

        # Return the original response to the client.
        return response

    except Exception:
        # Record unexpected server errors and re-raise the exception.
        metrics["failed_requests"] += 1
        logger.exception("Unexpected error while processing request")
        raise


# --------------------------------------------------
# 7. Health-check endpoint
# --------------------------------------------------

@app.get("/health")
def health_check():
    # Return a simple response to confirm the API is available.
    return {
        "status": "healthy",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


# --------------------------------------------------
# 8. Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(patient: PatientData):
    # Convert validated input into a one-row DataFrame.
    # Column names are important because the model uses these features.
    patient_df = pd.DataFrame([patient.model_dump()])

    # Generate the predicted class: 0 or 1.
    prediction = int(model.predict(patient_df)[0])

    # Estimate the model's probability for class 1.
    probability = float(model.predict_proba(patient_df)[0, 1])

    # Update monitoring counters after a successful prediction.
    metrics["successful_predictions"] += 1
    metrics[f"prediction_class_{prediction}"] += 1

    # Log the output class, but do not log the patient's feature values.
    logger.info(
        "prediction_completed class=%s probability_class_1=%.4f",
        prediction,
        probability,
    )

    # Return the prediction in JSON format.
    return {
        "prediction": prediction,
        "probability_class_1": round(probability, 4),
    }


# --------------------------------------------------
# 9. Basic monitoring endpoint
# --------------------------------------------------

@app.get("/metrics")
def get_metrics():
    # Return the monitoring counters as JSON.
    return {
        "service": "heart_disease_api",
        "metrics": metrics,
    }
