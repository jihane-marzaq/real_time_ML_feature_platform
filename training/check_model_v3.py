import mlflow
from config import MLFLOW_TRACKING_URI

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

MODEL_URI = "models:/fraud_detection_model/5"

print("Loading model:")
print(MODEL_URI)

model = mlflow.xgboost.load_model(MODEL_URI)

print("\nModel loaded successfully!")

print("\nFeature names:")
print(model.feature_names_in_)

print("\nNumber of features:")
print(len(model.feature_names_in_))

print("\nFeature types:")
print(model.feature_types)