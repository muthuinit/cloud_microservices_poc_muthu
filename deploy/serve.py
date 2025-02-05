import os
import joblib
from flask import Flask, request, jsonify

# Load the model
MODEL_PATH = "/model/model.pkl"
model = joblib.load(MODEL_PATH)

# Initialize Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    # Get input data
    data = request.get_json(force=True)
    instances = data["instances"]

    # Make predictions
    predictions = model.predict(instances)

    # Return predictions
    return jsonify({"predictions": predictions.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)