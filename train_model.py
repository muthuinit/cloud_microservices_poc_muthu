import os
import logging
import joblib
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from google.cloud import bigquery, storage
from google.api_core.exceptions import GoogleAPICallError
from flask import Flask, request, jsonify

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "sixth-utility-449722-p8")
DATASET_ID = os.getenv("BQ_DATASET_ID", "housing_data")
TABLE_ID = os.getenv("BQ_TABLE_ID", "housing_table")
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "housing-data-bucket-poc")

# Initialize BigQuery and GCS clients
client_bq = bigquery.Client(project=PROJECT_ID)
client_gcs = storage.Client(project=PROJECT_ID)

# Initialize Flask app
app = Flask(__name__)

# Load data from BigQuery
def load_data_from_bq():
    """Load housing data from BigQuery."""
    try:
        query = f"""
            SELECT size, bedrooms, price
            FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
        """
        logger.info(f"Loading data from BigQuery: {PROJECT_ID}.{DATASET_ID}.{TABLE_ID}")
        data = client_bq.query(query).to_dataframe()
        logger.info(f"Successfully loaded {len(data)} rows.")
        return data
    except GoogleAPICallError as e:
        logger.error(f"Failed to load data from BigQuery: {e}")
        raise

# Train the model
def train_model(data):
    """Train a RandomForestRegressor model."""
    try:
        # Define features (X) and target (y)
        X = data[["size", "bedrooms"]]
        y = data["price"]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and train the RandomForestRegressor model
        logger.info("Training RandomForestRegressor model...")
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        logger.info(f"Model training completed. MSE: {mse}")

        return model
    except Exception as e:
        logger.error(f"Failed to train model: {e}")
        raise

# Save the trained model to Google Cloud Storage
def save_model_to_gcs(model):
    """Save the trained model to Google Cloud Storage."""
    try:
        # Save the trained model to a local file
        model_filename = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        joblib.dump(model, model_filename)

        # Upload the model file to Google Cloud Storage
        bucket = client_gcs.bucket(BUCKET_NAME)
        blob = bucket.blob(f"models/{model_filename}")
        blob.upload_from_filename(model_filename)

        logger.info(f"Model successfully saved to GCS at gs://{BUCKET_NAME}/models/{model_filename}")
    except GoogleAPICallError as e:
        logger.error(f"Failed to save model to GCS: {e}")
        raise

# Define prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to make predictions using the trained model."""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    data = request.get_json(force=True)
    size = data['size']
    bedrooms = data['bedrooms']
    
    # Make prediction
    prediction = model.predict([[size, bedrooms]])
    return jsonify({"predicted_price": prediction[0]})

# Main entry point
if __name__ == "__main__":
    try:
        # Load data from BigQuery
        data = load_data_from_bq()

        # Train the model
        model = train_model(data)

        # Save the trained model to Cloud Storage
        save_model_to_gcs(model)

        # Start Flask app and bind to port 8080
        port = int(os.environ.get("PORT", 8080))  # Default to 8080
        app.run(host="0.0.0.0", port=port)  # Listen on all network interfaces
    except Exception as e:
        logger.error(f"Script failed: {e}")
