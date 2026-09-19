# feature_store/feature_repo/feature_views.py
from datetime import timedelta
from feast import FeatureView, Field, BigQuerySource
from feast.types import Float64, Int64, String
from entities import card

# Source BigQuery
card_features_source = BigQuerySource(
    name="card_features_source",
    table="real-time-ml-feature-platform.fraud_detection.feat_transaction_features",
    timestamp_field="event_timestamp",
)

# Feature View alignée exactement sur la table BigQuery
card_features_view = FeatureView(
    name="card_features",
    entities=[card],
    ttl=timedelta(days=7),
    schema=[
        # Features temporelles
        Field(name="transaction_hour", dtype=Int64),
        Field(name="transaction_dow", dtype=Int64),

        # Features montant
        Field(name="transaction_amt", dtype=Float64),
        Field(name="transaction_amt_log", dtype=Float64),

        # Features catégorielles & produit
        Field(name="card_network", dtype=String),
        Field(name="card_type", dtype=String),
        Field(name="product_cd", dtype=String),
        Field(name="purchaser_email_domain", dtype=String),
        Field(name="recipient_email_domain", dtype=String),

        # Features de comptage
        Field(name="card_txn_count_24h", dtype=Int64),
        Field(name="card_txn_count_1h", dtype=Int64),
    ],
    source=card_features_source,
    online=True,
)