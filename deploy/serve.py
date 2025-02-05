import os
import joblib
import logging
from google.cloud import storage

# Set logging
logging.basicConfig(level=logging.INFO)

# Model parameters
MODEL_BUCKET = os.getenv("MODEL_BUCKET", "housing-data-bucket-poc")
MODEL_FILE = os.getenv("MODEL_FILE", "model_registry/model.pkl")
LOCAL_MODEL_PATH = "/model/model.pkl"

def download_model():
    """Download the trained model from GCS at runtime."""
    if not os.path.exists("/model"):
        os.makedirs("/model")
    try:
        logging.info(f"Downloading model from gs://{MODEL_BUCKET}/{MODEL_FILE}")
        client = storage.Client()
        bucket = client.bucket(MODEL_BUCKET)
        blob = bucket.blob(MODEL_FILE)
        blob.download_to_filename(LOCAL_MODEL_PATH)
        logging.info("Model downloaded successfully!")
        return joblib.load(LOCAL_MODEL_PATH)
    except Exception as e:
        logging.error(f"Failed to download model: {e}")
        return None

# Download the model when the container starts
model = download_model()
if model is None:
    raise RuntimeError("Failed to load model. Check GCS path and credentials.")
