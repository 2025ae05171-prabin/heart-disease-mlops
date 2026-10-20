# Heart Disease Prediction API

## Contents
- api.py: FastAPI application
- heart_disease_model.joblib: trained preprocessing and model pipeline
- requirements.txt: pinned Python dependencies
- Dockerfile: container build instructions

## Build the Docker image
docker build -t heart-disease-api .

## Run the container
docker run --rm -p 8000:8000 heart-disease-api

## Test the health endpoint
Open http://localhost:8000/health

## Open interactive API documentation
Open http://localhost:8000/docs

## Prediction
Send a POST request to /predict with all 13 input features as JSON.
The API returns a binary prediction and the model's estimated probability
for class 1. This project is an educational demonstration, not a
clinically validated diagnostic system.
