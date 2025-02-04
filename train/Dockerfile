# Use a newer Python base image
FROM python:3.9-slim

# Install SSL certificates
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Copy the training script and requirements
COPY train_model.py /train_model.py
COPY requirements.txt /requirements.txt

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r /requirements.txt

# Set the entrypoint
ENTRYPOINT ["python", "/train_model.py"]