import os
import logging
import pandas as pd
from google.cloud import bigquery, storage
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GCP configuration
GCP_PROJECT_ID = os.getenv("sixth-utility-449722-p8")
GCS_BUCKET_NAME = os.getenv("housing-data-bucket-poc")
BQ_DATASET = os.getenv("sixth-utility-449722-p8.housing_data")
MODEL_FILENAME = "house_price_model.pkl"

def load_data_from_bigquery():
    """Load housing data from BigQuery."""
    client = bigquery.Client(project=GCP_PROJECT_ID)
    query = f"""
        SELECT *
        FROM `{BQ_DATASET}.housing_table`
    """
    logger.info("Loading housing data from BigQuery...")
    df = client.query(query).to_dataframe()
    return df

def preprocess_data(df):
    """Preprocess the housing data."""
    logger.info("Preprocessing data...")
    # Drop rows with missing values
    df = df.dropna()
    # Separate features and target
    X = df.drop("price", axis=1)  # Assuming 'price' is the target column
    y = df["price"]
    return X, y

def train_model(X, y):
    """Train a regression model for house price prediction."""
    logger.info("Training model...")
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logger.info(f"Model Mean Squared Error: {mse:.2f}")
    logger.info(f"Model R^2 Score: {r2:.2f}")

    return model

def save_model_to_gcs(model, filename):
    """Save the trained model to Google Cloud Storage."""
    client = storage.Client(project=GCP_PROJECT_ID)
    bucket = client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(filename)

    # Save the model locally first
    local_path = f"/tmp/{filename}"
    joblib.dump(model, local_path)

    # Upload to GCS
    logger.info(f"Saving model to GCS: gs://{GCS_BUCKET_NAME}/housing_data.csv")
    blob.upload_from_filename(local_path)

def main():
    # Load data from BigQuery
    df = load_data_from_bigquery()

    # Preprocess the data
    X, y = preprocess_data(df)

    # Train the model
    model = train_model(X, y)

    # Save the model to GCS
    save_model_to_gcs(model, MODEL_FILENAME)

    logger.info("House price prediction model training and saving completed!")

if __name__ == "__main__":
    main()