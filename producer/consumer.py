# producer/consumer.py

import json
from confluent_kafka import Consumer, KafkaError
from config import KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC

# S'assure qu'on passe bien une chaîne de caractères (ex: "localhost:9092")
if isinstance(KAFKA_BOOTSTRAP_SERVERS, list):
    BOOTSTRAP_SERVERS = ','.join(KAFKA_BOOTSTRAP_SERVERS)
else:
    BOOTSTRAP_SERVERS = str(KAFKA_BOOTSTRAP_SERVERS)


def create_consumer():
    conf = {
        'bootstrap.servers': BOOTSTRAP_SERVERS,
        'group.id': 'fraud-detector-group',
        'auto.offset.reset': 'earliest'
    }
    
    consumer = Consumer(conf)
    consumer.subscribe([KAFKA_TOPIC])
    return consumer


def main():
    print(f"[CONSUMER] Connecting to Kafka at {BOOTSTRAP_SERVERS}")
    print(f"[CONSUMER] Listening on topic: {KAFKA_TOPIC}")

    consumer = create_consumer()
    received = 0
    fraud_count = 0

    try:
        while True:
            msg = consumer.poll(timeout=1.0)

            if msg is None:
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    print(f"[ERROR] Kafka consumer error: {msg.error()}")
                    break

            # Décodage JSON
            value_str = msg.value().decode('utf-8')
            transaction = json.loads(value_str)

            received += 1

            is_fraud = transaction.get('isFraud', 0) == 1
            if is_fraud:
                fraud_count += 1
                print(f"  ⚠ FRAUD | ID: {transaction.get('TransactionID')} "
                      f"| Amount: ${transaction.get('TransactionAmt', 0):.2f} "
                      f"| Offset: {msg.offset()}")

            if received % 100 == 0:
                rate = (fraud_count / received) * 100
                print(f"[CONSUMER] Received: {received} | Fraud: {fraud_count} ({rate:.1f}%)")

    except KeyboardInterrupt:
        print("\n[CONSUMER] Arrêt du consommateur...")
    finally:
        consumer.close()


if __name__ == '__main__':
    main()