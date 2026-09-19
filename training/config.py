# training/config.py
import os

# Configuration GCP & BigQuery
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID") or "real-time-ml-feature-platform"
BQ_DATASET = "fraud_detection"
BQ_TABLE = f"{GCP_PROJECT_ID}.{BQ_DATASET}.feat_transaction_features"

# Target & Features
TARGET_COL = "is_fraud"

FEATURE_COLS = [
    # Temporel
    "transaction_hour",
    "transaction_dow",

    # Montant
    "transaction_amt_log",

    # Comportement / catégoriel
    "card_network",
    "card_type",
    "product_cd",
    "purchaser_email_domain",
    "recipient_email_domain",

    # Vélocité
    "card_txn_count_24h",
    "card_txn_count_1h",
]

# MLflow Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI") or "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT = "fraud-detection-xgboost"
MLFLOW_EXPERIMENT_NAME = MLFLOW_EXPERIMENT
MODEL_NAME = "fraud_detection_model"

# Paramètres généraux
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Paramètres XGBoost
XGBOOST_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "scale_pos_weight": 10,
    "random_state": 42,
    "eval_metric": "aucpr",
}