# training/data_loader.py
# Charge les données d'entraînement depuis BigQuery directement
# (contournement du problème offline Feast timestamp)

import pandas as pd
from google.cloud import bigquery
import os
from config import GCP_PROJECT_ID, BQ_TABLE, FEATURE_COLS, TARGET_COL


def load_training_data() -> pd.DataFrame:
    """
    Charge les features depuis BigQuery.
    On lit directement la table dbt feat_transaction_features.

    Pourquoi pas Feast offline store ici ?
    Le problème de timestamp (données 2017-2018 vs fenêtre actuelle)
    rend le offline retrieval Feast vide sur notre dataset.
    En production avec des données live, Feast offline fonctionnerait
    parfaitement — c'est une contrainte du dataset historique Kaggle.
    On documente ce compromis dans le rapport.
    """
    print(f"[DATA LOADER] Connecting to BigQuery...")
    client = bigquery.Client(project=GCP_PROJECT_ID)

    cols = ", ".join(FEATURE_COLS + [TARGET_COL, "transaction_id", "event_timestamp"])
    query = f"""
        SELECT {cols}
        FROM `{BQ_TABLE}`
        WHERE {TARGET_COL} IS NOT NULL
        ORDER BY event_timestamp
    """

    print(f"[DATA LOADER] Running query on {BQ_TABLE}...")
    df = client.query(query).to_dataframe()

    print(f"[DATA LOADER] Loaded {len(df):,} rows")
    print(f"[DATA LOADER] Fraud rate: {df[TARGET_COL].mean()*100:.2f}%")
    print(f"[DATA LOADER] Features: {FEATURE_COLS}")

    return df


def check_data_quality(df: pd.DataFrame) -> None:
    """
    Validation basique avant entraînement.
    Fail fast : mieux vaut planter ici qu'avoir un modèle silencieusement mauvais.
    """
    print("\n[DATA QUALITY] Running checks...")

    # Check 1 : pas de target manquant
    assert df[TARGET_COL].isna().sum() == 0, \
        f"Target has {df[TARGET_COL].isna().sum()} nulls"

    # Check 2 : taux de fraude raisonnable (entre 1% et 30%)
    fraud_rate = df[TARGET_COL].mean()
    assert 0.01 < fraud_rate < 0.30, \
        f"Fraud rate {fraud_rate:.2%} looks wrong"

    # Check 3 : features disponibles
    missing_features = [f for f in FEATURE_COLS if f not in df.columns]
    assert len(missing_features) == 0, \
        f"Missing features: {missing_features}"

    # Check 4 : volume suffisant
    assert len(df) > 1000, \
        f"Not enough data: {len(df)} rows"

    # Report valeurs manquantes par feature
    null_counts = df[FEATURE_COLS].isna().sum()
    null_features = null_counts[null_counts > 0]
    if len(null_features) > 0:
        print(f"[DATA QUALITY] Features with nulls (will be imputed):")
        for feat, count in null_features.items():
            print(f"  {feat}: {count:,} nulls ({count/len(df)*100:.1f}%)")

    print(f"[DATA QUALITY] ✅ All checks passed — {len(df):,} rows ready")