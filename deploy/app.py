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
