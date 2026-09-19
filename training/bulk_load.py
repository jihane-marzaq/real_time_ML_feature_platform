import os
import pandas as pd
from google.cloud import bigquery

PROJECT_ID = "real-time-ml-feature-platform"
DATASET_ID = "fraud_detection"
TABLE_ID = "raw_transactions"

TABLE_REF = f"real-time-ml-feature-platform.fraud_detection.raw_transactions"
CSV_PATH = "../data/train_transaction.csv"
TEMP_PARQUET = "temp_chunk.parquet"
CHUNK_SIZE = 100_000 

def bulk_load_to_bigquery():
    print(f"=== DEBUT DU CHARGEMENT BULK DEPUIS {CSV_PATH} ===")
    
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Le fichier {CSV_PATH} est introuvable.")

    client = bigquery.Client(project=PROJECT_ID)

    # 1. Supprimer l'ancienne table obsolète si elle existe
    print(f"Nettoyage : Suppression de l'ancienne table {TABLE_REF} (si elle existe)...")
    client.delete_table(TABLE_REF, not_found_ok=True)

    # 2. Chargement du CSV par morceaux
    csv_reader = pd.read_csv(CSV_PATH, chunksize=CHUNK_SIZE, low_memory=False)
    total_rows = 0

    for i, chunk in enumerate(csv_reader):
        print(f"-> Traitement du Chunk {i + 1} ({len(chunk)} lignes)...")

        # Conversion temporaire du chunk en format Parquet
        chunk.to_parquet(TEMP_PARQUET, index=False)

        # Configuration standard d'insertion
        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.PARQUET,
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND
        )

        # Envoi à BigQuery
        with open(TEMP_PARQUET, "rb") as source_file:
            job = client.load_table_from_file(source_file, TABLE_REF, job_config=job_config)
            job.result()  # Attendre la validation de l'insertion

        total_rows += len(chunk)
        print(f"   [OK] Chunk {i + 1} inséré avec succès.")

    # 3. Nettoyage du fichier Parquet temporaire
    if os.path.exists(TEMP_PARQUET):
        os.remove(TEMP_PARQUET)

    print(f"\n=== CHARGEMENT TERMINÉ AVEC SUCCÈS ===")
    print(f"Total des lignes injectées dans BigQuery : {total_rows}")

if __name__ == "__main__":
    bulk_load_to_bigquery()