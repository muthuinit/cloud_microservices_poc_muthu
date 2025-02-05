import xgboost as xgb
import numpy as np

# Create a simple XGBoost model
X = np.array([[1, 2], [2, 3], [3, 4]])
y = np.array([1, 2, 3])

dtrain = xgb.DMatrix(X, label=y)
params = {"objective": "reg:squarederror", "max_depth": 2}
model = xgb.train(params, dtrain, num_boost_round=10)

# Save the model
model.save_model("model.bst")