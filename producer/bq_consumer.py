# producer/bq_consumer.py

import json
import os
from confluent_kafka import Consumer, KafkaError
from google.cloud import bigquery
from config import KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC

# 1. Configuration GCP / BigQuery
# (S'assure de pointer vers les credentials si pas défini globalement)
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"

PROJECT_ID = "real-time-ml-feature-platform"
DATASET_ID = "fraud_detection"
TABLE_NAME = "raw_transactions"
TABLE_ID = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_NAME}"

BUFFER_SIZE = 10


def get_bootstrap_servers():
    """Formate correctement la chaîne pour confluent-kafka."""
    if isinstance(KAFKA_BOOTSTRAP_SERVERS, list):
        return ",".join(KAFKA_BOOTSTRAP_SERVERS)
    return str(KAFKA_BOOTSTRAP_SERVERS)


def ensure_bq_table(client):
    """Vérifie ou crée la table BigQuery."""
    dataset_ref = bigquery.DatasetReference(PROJECT_ID, DATASET_ID)
    table_ref = dataset_ref.table(TABLE_NAME)

    try:
        client.get_table(table_ref)
        print(f"[BQ] Table already exists: {TABLE_ID}")
    except Exception:
        # Si la table n'existe pas, BigQuery la créera au premier batch load
        print(f"[BQ] Table {TABLE_ID} ready for creation on first load.")


def flush_buffer(client, buffer):
    """Envoie le buffer vers BigQuery via Batch Loading (100% compatible Free Tier)."""
    if not buffer:
        return 0

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        autodetect=True,
        schema_update_options=[
            bigquery.SchemaUpdateOption.ALLOW_FIELD_RELAXATION
        ]
    )

    try:
        load_job = client.load_table_from_json(
            buffer,
            TABLE_ID,
            job_config=job_config
        )
        load_job.result()  # Attend l'exécution du job

        count = len(buffer)
        print(f"[BQ] Successfully inserted batch of {count} rows.")
        buffer.clear()
        return count
    except Exception as e:
        print(f"[BQ ERROR] Batch insert failed: {e}")
        return 0


def create_kafka_consumer():
    """Initialise le consommateur Confluent Kafka."""
    bootstrap = get_bootstrap_servers()
    conf = {
        'bootstrap.servers': bootstrap,
        'group.id': 'bq-consumer-group-v2',
        'auto.offset.reset': 'earliest'
    }
    consumer = Consumer(conf)
    consumer.subscribe([KAFKA_TOPIC])
    return consumer


def main():
    print("[BQ CONSUMER] Starting")
    print(f"[BQ CONSUMER] Target: {TABLE_ID}")
    print(f"[BQ CONSUMER] Buffer size: {BUFFER_SIZE}")

    # Initialisation du client BigQuery
    bq_client = bigquery.Client(project=PROJECT_ID)
    ensure_bq_table(bq_client)

    # Initialisation de Kafka Consumer
    consumer = create_kafka_consumer()
    print("[BQ CONSUMER] Connected to Kafka, listening...")

    buffer = []
    total_flushed = 0

    try:
        while True:
            msg = consumer.poll(timeout=1.0)

            if msg is None:
                # Si rien ne passe pendant le timeout et qu'on a des éléments en attente dans le buffer, on flush
                if buffer:
                    flushed = flush_buffer(bq_client, buffer)
                    total_flushed += flushed
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    print(f"[BQ CONSUMER ERROR] Kafka error: {msg.error()}")
                    break

            # Décodage JSON
            transaction = json.loads(msg.value().decode('utf-8'))
            buffer.append(transaction)

            # Flush dès que le buffer atteint BUFFER_SIZE
            if len(buffer) >= BUFFER_SIZE:
                flushed = flush_buffer(bq_client, buffer)
                total_flushed += flushed
                print(f"[BQ CONSUMER] Total rows written to BigQuery: {total_flushed}")

    except KeyboardInterrupt:
        print("\n[BQ CONSUMER] Stopping...")
        if buffer:
            flush_buffer(bq_client, buffer)
    finally:
        consumer.close()


if __name__ == '__main__':
    main()