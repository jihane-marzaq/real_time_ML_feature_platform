from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# COLONNES NUMÉRIQUES
# ============================================================

NUMERIC_FEATURES = [
    "transaction_amt",
    "transaction_amt_log",
    "transaction_hour",
    "transaction_dow",
    "card_txn_count_24h",
    "card_txn_count_1h",
]


# ============================================================
# COLONNES CATÉGORIELLES
# ============================================================

CATEGORICAL_FEATURES = [
    "card_network",
    "card_type",
    "product_cd",
    "purchaser_email_domain",
    "recipient_email_domain",
]


# ============================================================
# CRÉATION DU PREPROCESSOR
# ============================================================

def create_preprocessor():

    # --------------------------------------------------------
    # Pipeline numérique
    # --------------------------------------------------------

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
        ]
    )

    # --------------------------------------------------------
    # Pipeline catégoriel
    # --------------------------------------------------------

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="unknown",
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    # --------------------------------------------------------
    # Combinaison des deux pipelines
    # --------------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    return preprocessor