# producer/producer.py

import json
import time
import pandas as pd
from confluent_kafka import Producer
from config import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TOPIC,
    TRANSACTION_DELAY_SECONDS,
    DATA_PATH
)


def get_bootstrap_servers():
    """Formate correctement la chaîne pour confluent-kafka."""
    if isinstance(KAFKA_BOOTSTRAP_SERVERS, list):
        return ",".join(KAFKA_BOOTSTRAP_SERVERS)
    return str(KAFKA_BOOTSTRAP_SERVERS)


def delivery_report(err, msg):
    """
    Callback en cas d'erreur ou de succès d'envoi.
    Évite de faire crasher le script si un message échoue.
    """
    if err is not None:
        print(f"[ERROR] Delivery failed: {err}")


def create_producer():
    """
    Crée et configure le Producer Confluent Kafka.
    - 'acks': 'all' (1) garantit qu'aucune donnée n'est perdue.
    - 'retries': 3 retente en cas de micro-coupure réseau.
    """
    bootstrap = get_bootstrap_servers()
    conf = {
        'bootstrap.servers': bootstrap,
        'acks': 'all',
        'retries': 3,
        'client.id': 'fraud-transaction-producer'
    }
    return Producer(conf)


def load_dataset(path: str) -> pd.DataFrame:
    """
    Charge le dataset et sélectionne les colonnes utiles.
    """
    df = pd.read_csv(path)

    cols = [
        'TransactionID', 'TransactionDT', 'TransactionAmt',
        'ProductCD', 'card1', 'card4', 'card6',
        'P_emaildomain', 'R_emaildomain', 'isFraud'
    ]
    available = [c for c in cols if c in df.columns]
    return df[available]


def send_transaction(producer: Producer, transaction: dict):
    """
    Envoie une transaction dans Kafka avec Confluent-Kafka.
    
    Même clé (TransactionID) = même partition = ordre garanti.
    """
    key = str(transaction['TransactionID']).encode('utf-8')
    payload = json.dumps(transaction).encode('utf-8')

    producer.produce(
        topic=KAFKA_TOPIC,
        key=key,
        value=payload,
        callback=delivery_report
    )
    # Servir les callbacks en arrière-plan sans bloquer
    producer.poll(0)


def main():
    print(f"[PRODUCER] Starting — topic: {KAFKA_TOPIC}")
    print(f"[PRODUCER] Loading dataset from {DATA_PATH}")

    df = load_dataset(DATA_PATH)
    print(f"[PRODUCER] Loaded {len(df)} transactions")

    producer = create_producer()
    print(f"[PRODUCER] Connected to Kafka at {KAFKA_BOOTSTRAP_SERVERS}")

    sent = 0
    fraud_sent = 0

    try:
        for _, row in df.iterrows():
            transaction = row.dropna().to_dict()

            # Convertit les types numpy en types Python natifs
            transaction = {
                k: int(v) if hasattr(v, 'item') and isinstance(v.item(), int)
                else float(v) if hasattr(v, 'item') and isinstance(v.item(), float)
                else v
                for k, v in transaction.items()
            }

            send_transaction(producer, transaction)
            sent += 1

            if transaction.get('isFraud', 0) == 1:
                fraud_sent += 1

            # Log toutes les 100 transactions
            if sent % 100 == 0:
                print(f"[PRODUCER] Sent: {sent} | Fraud: {fraud_sent} ({fraud_sent/sent*100:.1f}%)")
                producer.flush()

            time.sleep(TRANSACTION_DELAY_SECONDS)

    except KeyboardInterrupt:
        print("\n[PRODUCER] Interrupted by user. Stopping...")
    finally:
        # Flush garantit que tous les messages en buffer sont envoyés
        producer.flush()
        print(f"[PRODUCER] Done. Total sent: {sent}")


if __name__ == '__main__':
    main()