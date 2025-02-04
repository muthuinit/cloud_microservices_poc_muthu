from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

# Load model from GCS during startup
MODEL_PATH = "/app/model.pkl"
MODEL_BUCKET = "housing-data-bucket-poc"
MODEL_BLOB = "models/latest_model.pkl"

app = Flask(__name__)

def load_model():
    """Load the latest model from GCS into the container."""
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(MODEL_BUCKET)
    blob = bucket.blob(MODEL_BLOB)
    blob.download_to_filename(MODEL_PATH)
    return joblib.load(MODEL_PATH)

model = load_model()

@app.route("/predict", methods=["POST"])
def predict():
    """Prediction endpoint."""
    try:
        data = request.json
        df = pd.DataFrame([data])
        prediction = model.predict(df)
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
