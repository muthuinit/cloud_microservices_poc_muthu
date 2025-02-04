import os
import logging
import joblib
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from google.cloud import bigquery, storage
from google.api_core.exceptions import GoogleAPICallError

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment Variables
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "sixth-utility-449722-p8")
DATASET_ID = os.getenv("BQ_DATASET_ID", "housing_data")
TABLE_ID = os.getenv("BQ_TABLE_ID", "housing_table")
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "housing-data-bucket-poc")
MODEL_BUCKET_PATH = f"gs://{BUCKET_NAME}/models"
MODEL_REGISTRY_PATH = f"gs://{BUCKET_NAME}/model_registry"

# Initialize Clients
client_bq = bigquery.Client()
client_gcs = storage.Client()

# Initialize Flask App
app = Flask(__name__)

def load_data_from_bq():
    """Load housing data from BigQuery."""
    query = f"""
        SELECT size, bedrooms, price
        FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
    """
    try:
        logger.info(f"Loading data from BigQuery table: {TABLE_ID}")
        return client_bq.query(query).to_dataframe()
    except GoogleAPICallError as e:
        logger.error(f"BigQuery Data Load Error: {e}")
        raise

def train_model(data):
    """Train a RandomForestRegressor model."""
    X = data[["size", "bedrooms"]]
    y = data["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    logger.info(f"Model Trained - MSE: {mse}")
    return model

def save_model_to_gcs(model):
    """Save trained model to GCS and Model Registry."""
    model_filename = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    local_path = f"/tmp/{model_filename}"
    joblib.dump(model, local_path)
    
    bucket = client_gcs.bucket(BUCKET_NAME)
    blob = bucket.blob(f"models/{model_filename}")
    blob.upload_from_filename(local_path)
    logger.info(f"Model saved to GCS: {MODEL_BUCKET_PATH}/{model_filename}")
    
    # Save to Model Registry
    registry_blob = bucket.blob(f"model_registry/{model_filename}")
    registry_blob.upload_from_filename(local_path)
    logger.info(f"Model also saved to Model Registry: {MODEL_REGISTRY_PATH}/{model_filename}")
    
    return model_filename

@app.route("/train", methods=["POST"])
def train():
    """HTTP Endpoint for training the model."""
    try:
        data = load_data_from_bq()
        model = train_model(data)
        model_filename = save_model_to_gcs(model)
        return jsonify({"message": "Model trained successfully", "model_filename": model_filename}), 200
    except Exception as e:
        logger.error(f"Training Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
