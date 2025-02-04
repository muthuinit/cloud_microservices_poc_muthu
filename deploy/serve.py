import os
import joblib
import json
import flask
import logging
from google.cloud import storage
from flask import Flask, request, jsonify

# Set up Flask app
app = Flask(__name__)

# Model parameters
MODEL_BUCKET = os.getenv("MODEL_BUCKET", "housing-data-bucket-poc")
MODEL_FILE = os.getenv("MODEL_FILE", "model_registry/model.pkl")
LOCAL_MODEL_PATH = "/tmp/model.pkl"

# Load the model from Google Cloud Storage
def download_model():
    """Download the trained model from GCS and load it into memory."""
    try:
        client = storage.Client()
        bucket = client.bucket(MODEL_BUCKET)
        blob = bucket.blob(MODEL_FILE)
        blob.download_to_filename(LOCAL_MODEL_PATH)
        model = joblib.load(LOCAL_MODEL_PATH)
        logging.info(f"Model loaded successfully from {MODEL_BUCKET}/{MODEL_FILE}")
        return model
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return None

# Load model at startup
model = download_model()
if model is None:
    raise RuntimeError("Failed to load model. Check GCS path and model format.")

@app.route("/predict", methods=["POST"])
def predict():
    """API endpoint to make predictions."""
    try:
        data = request.get_json()
        if not data or "instances" not in data:
            return jsonify({"error": "Missing 'instances' in request"}), 400
        
        instances = data["instances"]
        predictions = model.predict(instances).tolist()
        return jsonify({"predictions": predictions})

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
