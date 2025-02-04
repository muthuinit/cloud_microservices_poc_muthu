import os
import joblib
import numpy as np
from flask import Flask, request, jsonify
from google.cloud import bigquery
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Initialize Flask app
app = Flask(__name__)

# Load the trained model (if it already exists in the container)
model = None

# Load data from BigQuery
def load_data_from_bq():
    client = bigquery.Client()
    
    # Get Project ID from environment variable
    project_id = os.environ.get("GCP_PROJECT")  # Ensure this is set in your Cloud Run service
    
    query = f"""
        SELECT size, bedrooms, price
        FROM `sixth-utility-449722-p8.housing_data.housing_table`
    """
    return client.query(query).to_dataframe()

# Train and save the model (if not already trained and saved)
def train_model():
    data = load_data_from_bq()
    X = data[["size", "bedrooms"]]
    y = data["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    # Save the trained model using joblib
    joblib.dump(model, "model.pkl")
    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model training completed. MSE: {mse}")
    return model

# Load the model into memory when the app starts
def load_model():
    global model
    try:
        model = joblib.load("model.pkl")
        print("Model loaded successfully")
    except Exception as e:
        print("Model not found, training the model now...")
        model = train_model()

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model is not loaded"}), 500
    
    # Get input data from the request
    data = request.get_json(force=True)
    size = data['size']
    bedrooms = data['bedrooms']
    
    # Make prediction
    prediction = model.predict([[size, bedrooms]])
    
    # Return prediction as JSON
    return jsonify({"predicted_price": prediction[0]})

# Initialize the model on startup
load_model()

# Run Flask server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Default to 8080
    app.run(host='0.0.0.0', port=port)
