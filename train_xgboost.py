import os
import xgboost as xgb
from google.cloud import storage

# Define GCS bucket and model details
BUCKET_NAME = "housing-data-bucket-poc"
MODEL_DIR = "model_registry"
MODEL_FILENAME = "model.bst"
LOCAL_MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

# Ensure directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Sample training data (replace with your actual data)
X_train = [[1, 2], [2, 3], [3, 4], [4, 5]]
y_train = [10, 20, 30, 40]

# Train the model
model = xgb.XGBRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Save the model as model.bst (XGBoost format)
model.save_model(LOCAL_MODEL_PATH)
print(f"✅ Model saved locally at: {LOCAL_MODEL_PATH}")

# Upload the model to GCS
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)
blob = bucket.blob(f"{MODEL_DIR}/{MODEL_FILENAME}")
blob.upload_from_filename(LOCAL_MODEL_PATH)

print(f"✅ Model uploaded to GCS: gs://{BUCKET_NAME}/{MODEL_DIR}/{MODEL_FILENAME}")
