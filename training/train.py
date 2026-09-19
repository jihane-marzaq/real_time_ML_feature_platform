# training/train.py
# ============================================================
# PIPELINE PRINCIPAL D'ENTRAÎNEMENT
# Real-Time ML Feature Platform
#
# Pipeline :
# BigQuery
#    ↓
# Preprocessing
#    ↓
# Train/Test Split
#    ↓
# XGBoost
#    ↓
# Evaluation
#    ↓
# Artifacts
#    ↓
# MLflow
#    ↓
# Model Registry
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd

import mlflow
import mlflow.xgboost

from mlflow.models.signature import infer_signature

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)

from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 2. PROJECT IMPORTS
# ============================================================

from config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT,
    MODEL_NAME,
    FEATURE_COLS,
    TARGET_COL,
    XGBOOST_PARAMS,
    TEST_SIZE,
    RANDOM_STATE
)

from data_loader import (
    load_training_data,
    check_data_quality
)


# ============================================================
# 3. CONFIGURATION
# ============================================================

warnings.filterwarnings("ignore")


# Chemin du dossier training/
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# Dossier où seront sauvegardés les artifacts
ARTIFACTS_DIR = os.path.join(
    BASE_DIR,
    "artifacts"
)


# Création automatique du dossier
os.makedirs(
    ARTIFACTS_DIR,
    exist_ok=True
)


# ============================================================
# 4. PREPROCESSING
# ============================================================

def preprocess(df: pd.DataFrame):

    """
    Preprocessing minimal et reproductible.

    Étapes :
    1. Sélection des features
    2. Encodage des variables catégorielles
    3. Imputation des valeurs manquantes

    Important :
    Les mêmes transformations devront être reproduites
    au moment du serving pour éviter le data skew.
    """

    # --------------------------------------------------------
    # Sélection X et y
    # --------------------------------------------------------

    X = df[FEATURE_COLS].copy()

    y = df[TARGET_COL].copy()


    # --------------------------------------------------------
    # Identification des colonnes catégorielles
    # --------------------------------------------------------

    cat_cols = X.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()


    # Dictionnaire pour conserver les mappings
    encoders = {}


    # --------------------------------------------------------
    # Label Encoding
    # --------------------------------------------------------

    for col in cat_cols:

        le = LabelEncoder()


        # Remplacement des valeurs nulles
        # par "unknown"

        X[col] = (
            X[col]
            .fillna("unknown")
            .astype(str)
        )


        # Apprentissage du mapping

        X[col] = le.fit_transform(
            X[col]
        )


        # Sauvegarde du mapping

        encoders[col] = dict(
            zip(
                le.classes_,
                le.transform(le.classes_)
            )
        )


    # --------------------------------------------------------
    # Imputation des valeurs numériques
    # --------------------------------------------------------

    for col in X.columns:

        if X[col].isna().any():

            median_val = X[col].median()

            X[col] = X[col].fillna(
                median_val
            )


    return X, y, encoders


# ============================================================
# 5. CONFUSION MATRIX
# ============================================================

def plot_confusion_matrix(
    cm,
    path
):

    """
    Génère et sauvegarde la matrice de confusion.
    """

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )


    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "Not Fraud",
            "Fraud"
        ],
        yticklabels=[
            "Not Fraud",
            "Fraud"
        ],
        ax=ax
    )


    ax.set_xlabel(
        "Predicted"
    )

    ax.set_ylabel(
        "Actual"
    )

    ax.set_title(
        "Confusion Matrix"
    )


    plt.tight_layout()


    plt.savefig(
        path,
        dpi=150,
        bbox_inches="tight"
    )


    plt.close()


# ============================================================
# 6. FEATURE IMPORTANCE
# ============================================================

def plot_feature_importance(
    model,
    feature_names,
    path
):

    """
    Génère et sauvegarde
    la feature importance XGBoost.
    """

    importances = (
        model.feature_importances_
    )


    indices = np.argsort(
        importances
    )[::-1]


    fig, ax = plt.subplots(
        figsize=(10, 6)
    )


    ax.bar(
        range(len(importances)),
        importances[indices]
    )


    ax.set_xticks(
        range(len(importances))
    )


    ax.set_xticklabels(
        [
            feature_names[i]
            for i in indices
        ],
        rotation=45,
        ha="right"
    )


    ax.set_title(
        "Feature Importance (XGBoost)"
    )


    ax.set_ylabel(
        "Importance Score"
    )


    plt.tight_layout()


    plt.savefig(
        path,
        dpi=150,
        bbox_inches="tight"
    )


    plt.close()


# ============================================================
# 7. TRAINING FUNCTION
# ============================================================

def train():

    # ========================================================
    # STEP 1 — LOAD DATA
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "STEP 1 — LOADING DATA"
    )

    print(
        "=" * 60
    )


    df = load_training_data()


    # Vérification qualité

    check_data_quality(
        df
    )


    # ========================================================
    # STEP 2 — PREPROCESSING
    # ========================================================

    print(
        "\n[PREPROCESSING] Starting..."
    )


    X, y, encoders = preprocess(
        df
    )


    print(
        "[PREPROCESSING] Data preprocessing completed"
    )


    print(
        f"[PREPROCESSING] X shape: "
        f"{X.shape}"
    )


    print(
        f"[PREPROCESSING] Number of features: "
        f"{X.shape[1]}"
    )


    # ========================================================
    # STEP 3 — TRAIN / TEST SPLIT
    # ========================================================

    print(
        "\n[SPLIT] Creating train/test split..."
    )


    X_train, X_test, y_train, y_test = (
        train_test_split(

            X,
            y,

            test_size=TEST_SIZE,

            random_state=RANDOM_STATE,

            stratify=y
        )
    )


    print(
        f"\n[TRAIN] Train: "
        f"{len(X_train):,} rows | "
        f"Test: "
        f"{len(X_test):,} rows"
    )


    print(
        f"[TRAIN] Train fraud rate: "
        f"{y_train.mean() * 100:.2f}%"
    )


    print(
        f"[TRAIN] Test fraud rate: "
        f"{y_test.mean() * 100:.2f}%"
    )


    # ========================================================
    # STEP 4 — MLFLOW CONFIGURATION
    # ========================================================

    print(
        "\n[MLFLOW] Configuring MLflow..."
    )


    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )


    mlflow.set_experiment(
        MLFLOW_EXPERIMENT
    )


    # ========================================================
    # STEP 5 — START MLFLOW RUN
    # ========================================================

    with mlflow.start_run() as run:


        print(
            f"\n[MLFLOW] Run ID: "
            f"{run.info.run_id}"
        )


        # ====================================================
        # STEP 5.1 — LOG PARAMETERS
        # ====================================================

        mlflow.log_params(
            XGBOOST_PARAMS
        )


        mlflow.log_params({

            "test_size":
                TEST_SIZE,

            "random_state":
                RANDOM_STATE,

            "train_rows":
                len(X_train),

            "test_rows":
                len(X_test),

            "n_features":
                X.shape[1],

            "fraud_rate_train":
                float(y_train.mean()),

            "fraud_rate_test":
                float(y_test.mean())
        })


        # ====================================================
        # STEP 6 — TRAIN XGBOOST
        # ====================================================

        print(
            "\n[TRAIN] Training XGBoost..."
        )


        model = XGBClassifier(
            **XGBOOST_PARAMS
        )


        model.fit(

            X_train,

            y_train,

            eval_set=[
                (X_test, y_test)
            ],

            verbose=50
        )


        print(
            "[TRAIN] XGBoost training completed"
        )


        # ====================================================
        # STEP 7 — PREDICTIONS
        # ====================================================

        print(
            "\n[EVALUATION] Generating predictions..."
        )


        # Prediction 0 / 1

        y_pred = model.predict(
            X_test
        )


        # Probability of fraud

        y_proba = model.predict_proba(
            X_test
        )[:, 1]


        # ====================================================
        # STEP 8 — CLASSIFICATION METRICS
        # ====================================================

        # Métriques pour la classe positive = 1 = Fraud

        (
            fraud_precision,
            fraud_recall,
            fraud_f1,
            _

        ) = precision_recall_fscore_support(

            y_test,

            y_pred,

            average="binary",

            zero_division=0
        )


        # ====================================================
        # STEP 9 — AUC METRICS
        # ====================================================

        # ROC-AUC

        auc_roc = roc_auc_score(

            y_test,

            y_proba
        )


        # PR-AUC / Average Precision

        auc_pr = average_precision_score(

            y_test,

            y_proba
        )


        # ====================================================
        # STEP 10 — DISPLAY METRICS
        # ====================================================

        print(
            "\n" + "=" * 60
        )

        print(
            "MODEL METRICS"
        )

        print(
            "=" * 60
        )


        print(
            f"AUC-ROC  : "
            f"{auc_roc:.4f}"
        )


        print(
            f"AUC-PR   : "
            f"{auc_pr:.4f}"
        )


        print(
            f"Precision: "
            f"{fraud_precision:.4f}"
        )


        print(
            f"Recall   : "
            f"{fraud_recall:.4f}"
        )


        print(
            f"F1       : "
            f"{fraud_f1:.4f}"
        )


        # ====================================================
        # STEP 11 — LOG METRICS TO MLFLOW
        # ====================================================

        mlflow.log_metrics({

            "auc_roc":
                float(auc_roc),

            "auc_pr":
                float(auc_pr),

            "fraud_precision":
                float(fraud_precision),

            "fraud_recall":
                float(fraud_recall),

            "fraud_f1":
                float(fraud_f1)
        })


        print(
            "\n[MLFLOW] Metrics logged successfully"
        )


        # ====================================================
        # STEP 12 — CONFUSION MATRIX
        # ====================================================

        cm = confusion_matrix(

            y_test,

            y_pred
        )


        confusion_path = os.path.join(

            ARTIFACTS_DIR,

            "confusion_matrix.png"
        )


        plot_confusion_matrix(

            cm,

            confusion_path
        )


        mlflow.log_artifact(
            confusion_path
        )


        print(
            f"[ARTIFACT] Confusion matrix: "
            f"{confusion_path}"
        )


        # ====================================================
        # STEP 13 — FEATURE IMPORTANCE
        # ====================================================

        feature_importance_path = os.path.join(

            ARTIFACTS_DIR,

            "feature_importance.png"
        )


        plot_feature_importance(

            model,

            FEATURE_COLS,

            feature_importance_path
        )


        mlflow.log_artifact(

            feature_importance_path
        )


        print(
            f"[ARTIFACT] Feature importance: "
            f"{feature_importance_path}"
        )


        # ====================================================
        # STEP 14 — CLASSIFICATION REPORT
        # ====================================================

        report_path = os.path.join(

            ARTIFACTS_DIR,

            "classification_report.txt"
        )


        report_str = classification_report(

            y_test,

            y_pred,

            target_names=[
                "Not Fraud",
                "Fraud"
            ],

            zero_division=0
        )


        with open(

            report_path,

            "w",

            encoding="utf-8"

        ) as f:

            f.write(
                report_str
            )


        mlflow.log_artifact(
            report_path
        )


        print(
            f"[ARTIFACT] Classification report: "
            f"{report_path}"
        )


        # ====================================================
        # STEP 15 — MLFLOW MODEL SIGNATURE
        # ====================================================

        print(
            "\n[MLFLOW] Creating model signature..."
        )


        # La signature décrit :
        #
        # Input :
        # les 11 features du modèle
        #
        # Output :
        # prédiction 0 ou 1

        signature = infer_signature(

            X_train,

            model.predict(
                X_train
            )
        )


        # Exemple d'entrée

        input_example = (
            X_train.iloc[:3]
        )


        # ====================================================
        # STEP 16 — LOG MODEL
        # ====================================================

        print(
            "[MLFLOW] Logging XGBoost model..."
        )


        mlflow.xgboost.log_model(

            model,

            artifact_path="model",

            signature=signature,

            input_example=input_example,

            registered_model_name=MODEL_NAME
        )


        # ====================================================
        # STEP 17 — SUCCESS
        # ====================================================

        print(
            "\n" + "=" * 60
        )

        print(
            "[MLFLOW] ✅ Run logged successfully"
        )

        print(
            f"[MLFLOW] Model registered as: "
            f"'{MODEL_NAME}'"
        )

        print(
            f"[MLFLOW] Run ID: "
            f"{run.info.run_id}"
        )

        print(
            "=" * 60
        )


        # ====================================================
        # RETURN
        # ====================================================

        return (

            run.info.run_id,

            auc_roc
        )


# ============================================================
# 8. MAIN
# ============================================================

if __name__ == "__main__":


    run_id, auc = train()


    print(
        "\n" + "=" * 60
    )


    print(
        f"Run ID : {run_id}"
    )


    print(
        f"AUC-ROC: {auc:.4f}"
    )


    # Objectif actuel du projet

    if auc >= 0.90:

        print(
            "✅ Objectif AUC-ROC >= 0.90 atteint"
        )

    else:

        print(
            f"⚠️ AUC-ROC {auc:.4f} "
            f"< 0.90 — modèle à améliorer"
        )


    print(
        "=" * 60
    )