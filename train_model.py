import os
import logging
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from google.cloud import bigquery, storage

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment Variables
PROJECT_ID = "sixth-utility-449722-p8"
DATASET_ID = "housing_data"
TABLE_ID = "housing_table"
BUCKET_NAME = "housing-data-bucket-poc"
MODEL_DIR = "/tmp"
MODEL_PATH = f"{MODEL_DIR}/model.pkl"

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
    except Exception as e:
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
    """Save trained model to GCS."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    
    # Upload to GCS
    bucket = client_gcs.bucket(BUCKET_NAME)
    blob = bucket.blob("model_registry/model.pkl")
    blob.upload_from_filename(MODEL_PATH)
    logger.info(f"Model saved to GCS: gs://{BUCKET_NAME}/model_registry/model.pkl")

if __name__ == "__main__":
    data = load_data_from_bq()
    model = train_model(data)
    save_model_to_gcs(model)