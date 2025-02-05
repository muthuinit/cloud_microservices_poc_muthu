# Use an official Python image
FROM python:3.9-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    lsb-release && \
    curl -sSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    echo "deb https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    apt-get update && apt-get install -y google-cloud-sdk && \
    apt-get clean

# Install Python dependencies
COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /requirements.txt

# Set environment variables (for gsutil)
ENV PATH="/usr/lib/google-cloud-sdk/bin:$PATH"

# Create a directory for the model
RUN mkdir -p /model

# Copy the model serving script
COPY serve.py /serve.py

# Set the entrypoint
ENTRYPOINT ["python", "/serve.py"]
