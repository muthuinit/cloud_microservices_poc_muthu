# Use an official Python image
FROM python:3.9-slim

# Install dependencies
RUN pip install --upgrade pip && \
    pip install scikit-learn==1.0.2 google-cloud-storage flask

# Install Google Cloud SDK (for gsutil)
RUN apt-get update && \
    apt-get install -y curl && \
    curl -sSL https://sdk.cloud.google.com | bash && \
    apt-get install -y google-cloud-sdk && \
    apt-get clean

# Set environment variables
ENV PATH="/root/google-cloud-sdk/bin:$PATH"

# Create a directory for the model
RUN mkdir -p /model

# Copy the model serving script
COPY serve.py /serve.py

# Set the entrypoint
ENTRYPOINT ["python", "/serve.py"]
