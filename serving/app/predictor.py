import json
import math
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.xgboost
import pandas as pd
from feast import FeatureStore

from app.config import (
    MLFLOW_TRACKING_URI,
    MODEL_URI,
    MODEL_VERSION,
    FEAST_REPO_PATH,
    DECISION_THRESHOLD,
    FEATURE_COLS,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_PATH = (
    PROJECT_ROOT
    / "training"
    / "artifacts"
    / "preprocessing_metadata.json"
)


# TransactionDT du dataset IEEE-CIS :
# nombre de secondes depuis cette origine.
TRANSACTION_ORIGIN = datetime(
    2017,
    12,
    1,
    tzinfo=timezone.utc
)


class FraudPredictor:
    """
    Encapsule toute la logique de prédiction.

    Le modèle MLflow et Feast sont chargés une seule fois
    au démarrage de l'API.
    """

    def __init__(self):
        self.model = None
        self.store = None
        self.is_ready = False

        # Metadata du preprocessing utilisé pendant le training
        self.categorical_mappings = {}
        self.numeric_medians = {}

    # ========================================================
    # LOAD
    # ========================================================

    def load(self):
        """
        Charge :
        1. le modèle XGBoost depuis MLflow
        2. les metadata du preprocessing
        3. Feast
        """

        # ----------------------------------------------------
        # MLflow
        # ----------------------------------------------------

        print("[PREDICTOR] Loading model from MLflow...")

        mlflow.set_tracking_uri(
            MLFLOW_TRACKING_URI
        )

        self.model = mlflow.xgboost.load_model(
            MODEL_URI
        )

        print(
            f"[PREDICTOR] ✅ Model loaded: "
            f"{MODEL_URI}"
        )

        # ----------------------------------------------------
        # Preprocessing metadata
        # ----------------------------------------------------

        print(
            "[PREDICTOR] Loading preprocessing metadata..."
        )

        if not METADATA_PATH.exists():
            raise FileNotFoundError(
                f"Metadata file not found: "
                f"{METADATA_PATH}"
            )

        with open(
            METADATA_PATH,
            "r",
            encoding="utf-8"
        ) as f:
            metadata = json.load(f)

        metadata_version = str(
            metadata.get("model_version")
        )

        if metadata_version != str(MODEL_VERSION):
            raise ValueError(
                f"Metadata version {metadata_version} "
                f"does not match model version {MODEL_VERSION}"
            )

        metadata_features = metadata.get(
            "feature_cols",
            []
        )

        if metadata_features != FEATURE_COLS:
            raise ValueError(
                "Feature order mismatch between "
                "metadata and serving config."
            )

        self.categorical_mappings = metadata.get(
            "categorical_mappings",
            {}
        )

        self.numeric_medians = metadata.get(
            "numeric_medians",
            {}
        )

        print(
            "[PREDICTOR] ✅ Preprocessing metadata loaded"
        )

        # ----------------------------------------------------
        # Feast
        # ----------------------------------------------------

        print(
            "[PREDICTOR] Connecting to Feast..."
        )

        self.store = FeatureStore(
            repo_path=FEAST_REPO_PATH
        )

        print(
            "[PREDICTOR] ✅ Feast connected"
        )

        self.is_ready = True

    # ========================================================
    # CATEGORY ENCODING
    # ========================================================

    def _encode_category(
        self,
        column: str,
        value
    ) -> int:
        """
        Reproduit le LabelEncoder du training
        à partir du mapping sauvegardé.

        Exemple :
            visa -> 3
            credit -> 1
        """

        mapping = self.categorical_mappings.get(
            column
        )

        if not mapping:
            raise ValueError(
                f"No categorical mapping found "
                f"for column '{column}'"
            )

        # Même conversion que dans train.py :
        # fillna("unknown") puis astype(str)
        if value is None:
            value = "unknown"
        else:
            value = str(value)

        # Cas normal
        if value in mapping:
            return int(mapping[value])

        # ----------------------------------------------------
        # Cas particulier : transaction_amt_log
        # ----------------------------------------------------
        #
        # Cette colonne est catégorielle dans V5 car les
        # données BigQuery ont été vues comme object pendant
        # le preprocessing du training.
        #
        # Python peut afficher un float légèrement
        # différemment. On essaie donc une comparaison
        # numérique si la correspondance textuelle échoue.
        # ----------------------------------------------------

        if column == "transaction_amt_log":

            try:
                target_value = float(value)

                for category, encoded in mapping.items():

                    try:
                        category_value = float(
                            category
                        )

                        if math.isclose(
                            target_value,
                            category_value,
                            rel_tol=1e-12,
                            abs_tol=1e-12
                        ):
                            return int(encoded)

                    except (ValueError, TypeError):
                        continue

            except (ValueError, TypeError):
                pass

        # ----------------------------------------------------
        # Fallback unknown
        # ----------------------------------------------------

        if "unknown" in mapping:
            return int(mapping["unknown"])

        # transaction_amt_log n'a pas de catégorie "unknown"
        # car cette colonne a été traitée comme catégorielle
        # pendant le training de V5.
        if column == "transaction_amt_log":
            print(
                f"[PREDICTOR] Unseen transaction_amt_log "
                f"'{value}' -> fallback -1"
            )
            return -1

        raise ValueError(
            f"Unknown category '{value}' "
            f"for column '{column}' "
            f"and no 'unknown' mapping exists."
        )

    # ========================================================
    # FEATURE COMPUTATION
    # ========================================================

    def _compute_features_from_request(
        self,
        transaction: dict
    ) -> dict:
        """
        Calcule les 10 features attendues par V5.

        Sources :
        - features temporelles/montant : requête
        - catégories : requête
        - vélocité carte : Feast
        """

        # ----------------------------------------------------
        # Timestamp réel
        # ----------------------------------------------------

        actual_ts = (
            TRANSACTION_ORIGIN
            + pd.Timedelta(
                seconds=transaction["TransactionDT"]
            )
        )

        # ----------------------------------------------------
        # Features locales
        # ----------------------------------------------------

        transaction_amt_log = math.log1p(
            transaction["TransactionAmt"]
        )

        features = {
            "transaction_hour":
                actual_ts.hour,

            "transaction_dow":
                actual_ts.weekday(),

            "transaction_amt_log":
                transaction_amt_log,

            "card_network":
                transaction.get(
                    "card4"
                ) or "unknown",

            "card_type":
                transaction.get(
                    "card6"
                ) or "unknown",

            "product_cd":
                transaction.get(
                    "ProductCD"
                ) or "unknown",

            "purchaser_email_domain":
                transaction.get(
                    "P_emaildomain"
                ) or "unknown",

            "recipient_email_domain":
                transaction.get(
                    "R_emaildomain"
                ) or "unknown",
        }

        # ----------------------------------------------------
        # Feast
        # ----------------------------------------------------

        try:

            card_id = transaction["card1"]

            online = self.store.get_online_features(
                features=[
                    "card_features:card_txn_count_24h",
                    "card_features:card_txn_count_1h",
                ],
                entity_rows=[
                    {
                        "card_id": card_id
                    }
                ]
            ).to_dict()

            count_24h = online.get(
                "card_features:card_txn_count_24h",
                [None]
            )[0]

            count_1h = online.get(
                "card_features:card_txn_count_1h",
                [None]
            )[0]

            features[
                "card_txn_count_24h"
            ] = (
                0
                if count_24h is None
                else count_24h
            )

            features[
                "card_txn_count_1h"
            ] = (
                0
                if count_1h is None
                else count_1h
            )

        except Exception as e:

            print(
                f"[PREDICTOR] Feast fallback: {e}"
            )

            features[
                "card_txn_count_24h"
            ] = 0

            features[
                "card_txn_count_1h"
            ] = 0

        return features

    # ========================================================
    # PREPROCESSING
    # ========================================================

    def _preprocess(
        self,
        features: dict
    ) -> pd.DataFrame:

        # DataFrame dans l'ordre exact du modèle
        X = pd.DataFrame(
            [features]
        )[FEATURE_COLS].copy()

        # ----------------------------------------------------
        # Catégorielles → int
        # ----------------------------------------------------

        categorical_cols = [
            "transaction_amt_log",
            "card_network",
            "card_type",
            "product_cd",
            "purchaser_email_domain",
            "recipient_email_domain",
        ]

        for col in categorical_cols:

            X[col] = X[col].apply(
                lambda value: self._encode_category(
                    col,
                    value
                )
            )

        # ----------------------------------------------------
        # Numériques
        # ----------------------------------------------------

        numeric_cols = [
            "transaction_hour",
            "transaction_dow",
            "card_txn_count_24h",
            "card_txn_count_1h",
        ]

        for col in numeric_cols:

            X[col] = pd.to_numeric(
                X[col],
                errors="coerce"
            )

            if X[col].isna().any():

                median_value = (
                    self.numeric_medians.get(
                        col,
                        0
                    )
                )

                X[col] = X[col].fillna(
                    median_value
                )

        # ----------------------------------------------------
        # Types finaux
        # ----------------------------------------------------

        X = X.astype("int64")

        return X

    # ========================================================
    # RISK
    # ========================================================

    def _get_risk_level(
        self,
        score: float
    ) -> str:

        if score >= 0.6:
            return "HIGH"

        if score >= 0.3:
            return "MEDIUM"

        return "LOW"

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        transaction: dict
    ) -> dict:

        if not self.is_ready:
            raise RuntimeError(
                "Model not loaded"
            )

        # ----------------------------------------------------
        # 1. Features
        # ----------------------------------------------------

        features = (
            self._compute_features_from_request(
                transaction
            )
        )

        # ----------------------------------------------------
        # 2. Preprocessing
        # ----------------------------------------------------

        X = self._preprocess(
            features
        )

        # Debug utile pendant les tests
        print(
            "[PREDICTOR] Model input:"
        )
        print(X)
        print(
            "[PREDICTOR] dtypes:"
        )
        print(X.dtypes)

        # ----------------------------------------------------
        # 3. Prediction
        # ----------------------------------------------------

        score = float(
            self.model.predict_proba(
                X
            )[0][1]
        )

        is_fraud = (
            score >= DECISION_THRESHOLD
        )

        return {
            "score": score,
            "is_fraud": is_fraud,
            "features": features,
            "risk_level": self._get_risk_level(
                score
            ),
        }


# ============================================================
# GLOBAL INSTANCE
# ============================================================

predictor = FraudPredictor()