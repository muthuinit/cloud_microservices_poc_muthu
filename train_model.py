import os
import logging
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from google.cloud import storage

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment Variables
PROJECT_ID = "sixth-utility-449722-p8"
BUCKET_NAME = "housing-data-bucket-poc"
MODEL_DIR = "/tmp"
MODEL_PATH = f"{MODEL_DIR}/model.pkl"

# Initialize GCS Client
client_gcs = storage.Client()

def train_model():
    """Train a simple RandomForestRegressor model."""
    # Sample data
    data = {
        "size": [1000, 1500, 2000, 2500, 3000],
        "bedrooms": [2, 3, 4, 4, 5],
        "price": [300000, 400000, 500000, 550000, 600000],
    }
    df = pd.DataFrame(data)

    # Features and target
    X = df[["size", "bedrooms"]]
    y = df["price"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
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
    model = train_model()
    save_model_to_gcs(model)