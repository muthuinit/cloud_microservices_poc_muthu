# Use Python 3.10 slim image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080  # Cloud Run expects your app to listen on this port

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the Cloud Run port
EXPOSE 8080

# Use Gunicorn to serve the Flask app in production mode
CMD ["gunicorn", "-b", "0.0.0.0:8080", "train_model:app"]
