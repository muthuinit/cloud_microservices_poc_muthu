# Use Python 3.10 slim image
FROM python:3.10-slim

WORKDIR /app

# Copy the requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app and model file into the container
COPY train_model.py .
COPY model.pkl .

# Expose the correct port for Cloud Run
EXPOSE 8080

# Use Gunicorn to serve the Flask app in production mode
CMD ["gunicorn", "-b", "0.0.0.0:8080", "train_model:app"]
