import os
import logging
import joblib
import pandas as pd
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
MODEL_DIR = "/tmp"
MODEL_PATH = f"{MODEL_DIR}/model.pkl"  # Static filename
MODEL_BLOB = "models/latest_model.pkl"

# Initialize Clients
client_bq = bigquery.Client()
client_gcs = storage.Client()

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
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Save the model with a static filename
    local_path = f"{MODEL_DIR}/model.pkl"
    joblib.dump(model, local_path)
    
    # Upload to GCS
    bucket = client_gcs.bucket(BUCKET_NAME)
    blob = bucket.blob("models/model.pkl")
    blob.upload_from_filename(local_path)
    logger.info(f"Model saved to GCS: {MODEL_BUCKET_PATH}/model.pkl")
    
    # Save to Model Registry
    registry_blob = bucket.blob("model_registry/model.pkl")
    registry_blob.upload_from_filename(local_path)
    logger.info(f"Model also saved to Model Registry: {MODEL_REGISTRY_PATH}/model.pkl")

def load_model():
    """Load the latest model from GCS into the container."""
    os.makedirs(MODEL_DIR, exist_ok=True)  # Ensure directory exists
    bucket = client_gcs.bucket(BUCKET_NAME)
    blob = bucket.blob(MODEL_BLOB)
    
    if not blob.exists():
        raise FileNotFoundError(f"Model file not found in GCS: gs://{BUCKET_NAME}/{MODEL_BLOB}")
    
    blob.download_to_filename(MODEL_PATH)
    return joblib.load(MODEL_PATH)

if __name__ == "__main__":
    data = load_data_from_bq()
    model = train_model(data)
    save_model_to_gcs(model)
    logger.info("Model training completed. Saved as model.pkl")