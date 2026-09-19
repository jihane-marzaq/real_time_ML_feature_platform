import time
from datetime import datetime, timezone, timedelta
import pandas as pd
from feast import FeatureStore

store = FeatureStore(repo_path="feature_repo")

# ── TEST 1 : OFFLINE RETRIEVAL ─────────────────────────────────
print("=== TEST OFFLINE RETRIEVAL ===")

# Utilise la date d'aujourd'hui pour le point-in-time
now = datetime.now(timezone.utc)

entity_df = pd.DataFrame({
    "card_id": ["13926", "2755", "4497"],
    "event_timestamp": [now, now, now]
})

training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "card_features:transaction_amt",
        "card_features:transaction_amt_log",
        "card_features:card_txn_count_24h",
        "card_features:card_txn_count_1h",
    ]
).to_df()

print(training_df)
print(f"\nShape: {training_df.shape}")

# ── TEST 2 : ONLINE RETRIEVAL ──────────────────────────────────
print("\n=== TEST ONLINE RETRIEVAL ===")

card_ids = ["13926", "2755"]
features_to_fetch = [
    "card_features:transaction_amt",
    "card_features:transaction_amt_log",
    "card_features:card_txn_count_24h",
    "card_features:card_txn_count_1h",
]

# Warm-up (Échauffement de la connexion Redis)
_ = store.get_online_features(
    features=features_to_fetch,
    entity_rows=[{"card_id": card_ids[0]}]
)

# Mesure de la vraie latence en cache chaud
start = time.time()
online_features = store.get_online_features(
    features=features_to_fetch,
    entity_rows=[{"card_id": card_id} for card_id in card_ids]
).to_dict()
elapsed_ms = (time.time() - start) * 1000

print(f"Latence online retrieval (Warm connection) : {elapsed_ms:.2f}ms")
print(f"Features récupérées pour card_id={card_ids[0]} :")
for feat, values in online_features.items():
    print(f"  {feat}: {values[0]}")

# Assertion de latence temps réel (< 50ms acceptable en local)
assert elapsed_ms < 50, f"Latence trop élevée : {elapsed_ms:.2f}ms"
print("\n✅ Le test de latence passe parfaitement !")