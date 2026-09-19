# feature_store/feature_repo/feature_services.py
from feast import FeatureService
from feature_views import card_features_view

fraud_model_v1 = FeatureService(
    name="fraud_detection_v1",
    features=[
        card_features_view[[
            "transaction_hour",
            "transaction_dow",
            "transaction_amt",
            "transaction_amt_log",
            "card_network",
            "card_type",
            "product_cd",
            "purchaser_email_domain",
            "recipient_email_domain",
            "card_txn_count_24h",
            "card_txn_count_1h",
        ]]
    ],
    description="Features pour le modèle XGBoost fraud detection v1",
)