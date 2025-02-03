import os
from flask import Flask
import joblib

# Load the trained model
model = joblib.load("model.pkl")

# Create Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Model is ready!"

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data (JSON) from the request
    data = request.get_json(force=True)
    size = data['size']
    bedrooms = data['bedrooms']
    
    # Make prediction
    prediction = model.predict([[size, bedrooms]])
    
    # Return prediction as JSON
    return jsonify({"predicted_price": prediction[0]})

if __name__ == "__main__":
    # Flask listens on the correct port (8080 as required by Cloud Run)
    port = int(os.environ.get("PORT", 8080))  # Default to 8080
    app.run(host='0.0.0.0', port=port)
