# contiendra uniquement la description de la table BigQuery :
# producer/bq_schema.py
# Définit le schéma BigQuery de notre table
# Pourquoi séparer le schéma ? Si tu changes la structure,
# tu modifies un seul fichier — pas toute la logique

from google.cloud import bigquery

RAW_TRANSACTIONS_SCHEMA = [
    bigquery.SchemaField("TransactionID", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("TransactionDT", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("TransactionAmt", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("ProductCD", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("card1", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("card4", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("card6", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("P_emaildomain", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("R_emaildomain", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("isFraud", "INTEGER", mode="NULLABLE"),
    # Timestamp d'ingestion — crucial pour le debugging
    # et pour les features temporelles
    bigquery.SchemaField("ingested_at", "TIMESTAMP", mode="REQUIRED"),
]