# Use an official Python image
FROM python:3.9-slim

# Install dependencies
RUN pip install --upgrade pip && \
    pip install scikit-learn==1.0.2 google-cloud-storage flask

# Create a directory for the model
RUN mkdir -p /model

# Copy the model serving script and model file
COPY serve.py /serve.py
COPY model.pkl /model/model.pkl

# Set the entrypoint
ENTRYPOINT ["python", "/serve.py"]