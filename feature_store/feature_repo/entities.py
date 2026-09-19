# feature_store/feature_repo/entities.py
from feast import Entity, ValueType

card = Entity(
    name="card_id",
    value_type=ValueType.STRING,  # STRING pour correspondre à BigQuery
    description="Identifiant unique de la carte bancaire",
)