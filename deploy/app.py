import os
from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def home():
    return {"message": "House Price Prediction API is running!"}

@app.post("/predict/")
def predict(size: float, bedrooms: int):
    prediction = model.predict(np.array([[size, bedrooms]]))
    return {"predicted_price": prediction[0]}

# Explicitly listen on PORT 8080 for Cloud Run
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))  # Get PORT from env variable (default 8080)
    uvicorn.run(app, host="0.0.0.0", port=port)
