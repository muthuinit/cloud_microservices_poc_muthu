import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from google.cloud import bigquery, storage

# Initialize BigQuery client
client_bq = bigquery.Client()

# Initialize Cloud Storage client
client_gcs = storage.Client()

# Define the BigQuery dataset and table
dataset_id = "sixth-utility-449722-p8.housing_data"
table_id = f"{dataset_id}.housing_table"

# Define the Cloud Storage bucket name
bucket_name = "housing-data-bucket-poc"

# Load data from BigQuery
def load_data_from_bq():
    query = f"""
        SELECT size, bedrooms, price
        FROM `{table_id}`
    """
    # Query the BigQuery table and return the result as a pandas DataFrame
    data = client_bq.query(query).to_dataframe()
    return data

# Train the model
def train_model():
    data = load_data_from_bq()
    
    # Define features (X) and target (y)
    X = data[["size", "bedrooms"]]
    y = data["price"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Initialize and train the RandomForestRegressor model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model training completed. MSE: {mse}")

    return model

# Save the trained model to Google Cloud Storage
def save_model_to_gcs(model):
    # Save the trained model to a local file
    joblib.dump(model, "model.pkl")
    
    # Upload the model file to Google Cloud Storage
    bucket = client_gcs.bucket(bucket_name)
    blob = bucket.blob("models/model.pkl")
    blob.upload_from_filename("model.pkl")
    
    print(f"Model successfully saved to GCS at gs://{bucket_name}/models/model.pkl")

# Main function to train and save the model
if __name__ == "__main__":
    # Train the model
    model = train_model()

    # Save the trained model to Cloud Storage
    save_model_to_gcs(model)
