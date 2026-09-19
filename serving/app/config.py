

# MLflow
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAINING_DIR = PROJECT_ROOT / "training"

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{(TRAINING_DIR / 'mlflow.db').as_posix()}"
)

MODEL_NAME = "fraud_detection_model"
MODEL_VERSION = "5"

MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_VERSION}"

# Feast
FEAST_REPO_PATH     = str(
    PROJECT_ROOT / "feature_store" / "feature_repo"
)

# Seuil de décision — issu du threshold diagnostic
# On a choisi 0.30 : compromis Precision/Recall acceptable en banque
DECISION_THRESHOLD  = float(os.getenv("DECISION_THRESHOLD", "0.30"))

# Features — exactement les mêmes que training/config.py
# Si tu changes ici sans changer là : c'est du skew
FEATURE_COLS = [
    "transaction_hour",
    "transaction_dow",
    "transaction_amt_log",
    "card_network",
    "card_type",
    "product_cd",
    "purchaser_email_domain",
    "recipient_email_domain",
    "card_txn_count_24h",
    "card_txn_count_1h",
]