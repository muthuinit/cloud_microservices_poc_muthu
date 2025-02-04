# Use an official Python image
FROM python:3.9-slim

# Install dependencies
RUN pip install --upgrade pip && \
    pip install scikit-learn==1.0.2 google-cloud-storage

# Copy the model serving script
COPY serve.py /serve.py

# Set the entrypoint
ENTRYPOINT ["python", "/serve.py"]