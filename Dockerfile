
FROM python:3.11-slim

# Set the working directory inside the container.
WORKDIR /app

# Copy the dependency list and install the packages.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API code and saved model into the container.
COPY api.py .
COPY heart_disease_model.joblib .

# Document the port used by the API.
EXPOSE 8000

# Start the FastAPI server when the container launches.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
