import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from google.cloud import bigquery

# Load data from BigQuery
def load_data_from_bq():
    client = bigquery.Client()
    
    # Get Project ID from environment variable
    project_id = os.environ.get("GCP_PROJECT")  # Ensure this is set in your GitHub Actions workflow
    
    query = f"""
        SELECT size, bedrooms, price
        FROM `${{ secrets.GCP_PROJECT_ID }}.housing_data.housing_table`
    """
    return client.query(query).to_dataframe()

# Train model
def train_model():
    data = load_data_from_bq()
    X = data[["size", "bedrooms"]]
    y = data["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    return model, X_test, y_test

# Evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"MSE: {mse}")

# Save model to GCS
def save_model_to_gcs(model):
    import joblib
    from google.cloud import storage

    joblib.dump(model, "model.pkl")

    client = storage.Client()
    bucket = client.bucket("mlops-poc-bucket")
    blob = bucket.blob("models/model.pkl")
    blob.upload_from_filename("model.pkl")

if __name__ == "__main__":
    model, X_test, y_test = train_model()
    evaluate_model(model, X_test, y_test)
    save_model_to_gcs(model)
