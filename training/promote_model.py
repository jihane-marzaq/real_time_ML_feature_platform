import mlflow
from mlflow.tracking import MlflowClient

# Indiquer l'adresse du serveur MLflow qui tourne sur le port 5000
mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

MODEL_NAME = "fraud-xgboost"