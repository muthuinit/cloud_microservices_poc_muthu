# Use an official Python image
FROM python:3.9-slim

# Install dependencies
RUN pip install --upgrade pip && \
    pip install scikit-learn==1.0.2 google-cloud-storage flask

# Install gcloud CLI (for gsutil)
RUN apt-get update && \
    apt-get install -y curl && \
    curl -sSL https://sdk.cloud.google.com | bash && \
    export PATH=$PATH:/root/google-cloud-sdk/bin && \
    gcloud components install gsutil -q

# Create a directory for the model
RUN mkdir -p /model

# Copy the model serving script
COPY serve.py /serve.py

# Download the model file from GCS
RUN gsutil cp gs://housing-data-bucket-poc/model_registry/model.pkl /model/model.pkl

# Set the entrypoint
ENTRYPOINT ["python", "/serve.py"]