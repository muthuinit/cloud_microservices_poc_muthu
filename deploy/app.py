import os
from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()

# Load the model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def home():
    return {"message": "House Price Prediction API is running!"}

@app.post("/predict/")
def predict(size: float, bedrooms: int):
    prediction = model.predict(np.array([[size, bedrooms]]))
    return {"predicted_price": prediction[0]}

# Ensure the app listens on the correct port
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))  # Default to 8080 for Cloud Run
    uvicorn.run(app, host="0.0.0.0", port=port)
