import os
import joblib
import logging
from flask import Flask, request, jsonify
from google.cloud import storage

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

MODEL_BUCKET = os.getenv("MODEL_BUCKET", "housing-data-bucket-poc")
MODEL_FILE = os.getenv("MODEL_FILE", "model_registry/model.pkl")
LOCAL_MODEL_PATH = "/model/model.pkl"

def download_model():
    """Download the model from GCS at runtime"""
    if not os.path.exists("/model"):
        os.makedirs("/model")
    try:
        client = storage.Client()
        bucket = client.bucket(MODEL_BUCKET)
        blob = bucket.blob(MODEL_FILE)
        blob.download_to_filename(LOCAL_MODEL_PATH)
        logging.info(f"Model downloaded successfully from {MODEL_BUCKET}/{MODEL_FILE}")
        return joblib.load(LOCAL_MODEL_PATH)
    except Exception as e:
        logging.error(f"Failed to download model: {e}")
        return None

# Load model when the container starts
model = download_model()
if model is None:
    raise RuntimeError("Failed to load model. Check GCS path and credentials.")

@app.route("/predict", methods=["POST"])
def predict():
    """API endpoint to make predictions"""
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
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
