import os
import xgboost as xgb
from google.cloud import storage

# Define bucket and model details
BUCKET_NAME = "housing-data-bucket-poc"
MODEL_DIR = "model_registry"
MODEL_PATH = os.path.join(MODEL_DIR, "model.bst")

# Ensure directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Sample training data
X_train = [[1, 2], [2, 3], [3, 4], [4, 5]]
y_train = [10, 20, 30, 40]

# Train the model
model = xgb.XGBRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Save the model
model.save_model(MODEL_PATH)
print(f"Model saved locally: {MODEL_PATH}")

# Upload the **entire directory** to GCS
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)
blob = bucket.blob(f"{MODEL_DIR}/model.bst")
blob.upload_from_filename(MODEL_PATH)

print(f"Model uploaded to GCS: gs://{BUCKET_NAME}/{MODEL_DIR}/model.bst")
