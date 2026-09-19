# training/find_threshold.py
# ============================================================
# DIAGNOSTIC DU SEUIL DU MODÈLE V2
#
# IMPORTANT :
# - Aucun entraînement
# - Aucun model.fit()
# - Charge le modèle existant depuis MLflow
# - Reproduit exactement le preprocessing et le split
#   utilisés dans train.py
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd

import mlflow
import mlflow.xgboost

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve
)

from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


# ============================================================
# PROJECT IMPORTS
# ============================================================

from config import (
    MLFLOW_TRACKING_URI,
    MODEL_NAME,
    FEATURE_COLS,
    TARGET_COL,
    TEST_SIZE,
    RANDOM_STATE
)

from data_loader import (
    load_training_data,
    check_data_quality
)


# ============================================================
# CONFIGURATION
# ============================================================

# Modèle à diagnostiquer
MODEL_VERSION = "2"

# Seuils à tester
THRESHOLDS = [
    0.50,
    0.45,
    0.40,
    0.35,
    0.30,
    0.25,
    0.20,
    0.15,
    0.10,
    0.05
]

# Dossier artifacts
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

ARTIFACTS_DIR = os.path.join(
    BASE_DIR,
    "artifacts"
)

os.makedirs(
    ARTIFACTS_DIR,
    exist_ok=True
)


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess(df: pd.DataFrame):

    """
    Même preprocessing que train.py.

    IMPORTANT :
    On reproduit exactement la logique utilisée
    pendant l'entraînement du modèle V2.
    """

    X = df[FEATURE_COLS].copy()

    y = df[TARGET_COL].copy()

    print(
        f"[PREPROCESSING] Features utilisées : "
        f"{list(X.columns)}"
    )

    # --------------------------------------------------------
    # Encodage des variables catégorielles
    # --------------------------------------------------------

    cat_cols = X.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    print(
        f"[PREPROCESSING] Colonnes catégorielles : "
        f"{cat_cols}"
    )

    for col in cat_cols:

        le = LabelEncoder()

        X[col] = (
            X[col]
            .fillna("unknown")
            .astype(str)
        )

        X[col] = le.fit_transform(
            X[col]
        )

    # --------------------------------------------------------
    # Imputation des valeurs manquantes
    # --------------------------------------------------------

    for col in X.columns:

        if X[col].isna().any():

            median_val = X[col].median()

            X[col] = X[col].fillna(
                median_val
            )

    print(
        "[PREPROCESSING] ✅ Completed"
    )

    print(
        f"[PREPROCESSING] X shape : {X.shape}"
    )

    return X, y


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    """
    Charge la version 2 du modèle depuis MLflow.
    """

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    model_uri = (
        f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    )

    print(
        "\n[MLFLOW] Loading model..."
    )

    print(
        f"[MLFLOW] URI : {model_uri}"
    )

    model = mlflow.xgboost.load_model(
        model_uri
    )

    print(
        "[MLFLOW] ✅ Model loaded"
    )

    return model


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

def analyze_thresholds(
    y_test,
    y_proba
):

    """
    Analyse plusieurs seuils de décision.
    """

    print(
        "\n" + "=" * 100
    )

    print(
        "THRESHOLD ANALYSIS"
    )

    print(
        "=" * 100
    )

    print(
        f"{'Threshold':>10} "
        f"{'Precision':>12} "
        f"{'Recall':>10} "
        f"{'F1':>10} "
        f"{'Alerts':>10} "
        f"{'Detected':>10} "
        f"{'FP':>10} "
        f"{'FN':>10}"
    )

    print(
        "-" * 100
    )

    results = []

    total_fraud = int(
        y_test.sum()
    )

    for threshold in THRESHOLDS:

        # ----------------------------------------------------
        # Probability → class
        # ----------------------------------------------------

        y_pred = (
            y_proba >= threshold
        ).astype(int)

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        # ----------------------------------------------------
        # Confusion matrix
        # ----------------------------------------------------

        cm = confusion_matrix(
            y_test,
            y_pred
        )

        tn, fp, fn, tp = cm.ravel()

        # ----------------------------------------------------
        # Number of alerts
        # ----------------------------------------------------

        alerts = int(
            y_pred.sum()
        )

        detected = int(tp)

        results.append({

            "threshold":
                threshold,

            "precision":
                precision,

            "recall":
                recall,

            "f1":
                f1,

            "alerts":
                alerts,

            "detected":
                detected,

            "true_negatives":
                int(tn),

            "false_positives":
                int(fp),

            "false_negatives":
                int(fn),

            "true_positives":
                int(tp)
        })

        print(
            f"{threshold:>10.2f} "
            f"{precision:>12.3f} "
            f"{recall:>10.3f} "
            f"{f1:>10.3f} "
            f"{alerts:>10,} "
            f"{detected:>10,} "
            f"{fp:>10,} "
            f"{fn:>10,}"
        )

    print(
        "-" * 100
    )

    print(
        f"Total frauds in test set : "
        f"{total_fraud:,}"
    )

    return pd.DataFrame(
        results
    )


# ============================================================
# SAVE CSV
# ============================================================

def save_results(
    results
):

    path = os.path.join(
        ARTIFACTS_DIR,
        "threshold_analysis.csv"
    )

    results.to_csv(
        path,
        index=False
    )

    print(
        f"\n[ARTIFACT] Results saved :"
    )

    print(
        path
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    y_test,
    y_proba,
    threshold
):

    y_pred = (
        y_proba >= threshold
    ).astype(int)

    cm = confusion_matrix(
        y_test,
        y_pred
    )

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
        f"Confusion Matrix - "
        f"Threshold {threshold:.2f}"
    )

    plt.tight_layout()

    path = os.path.join(
        ARTIFACTS_DIR,
        f"confusion_matrix_threshold_{threshold:.2f}.png"
    )

    plt.savefig(
        path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"[ARTIFACT] Confusion matrix saved : "
        f"{path}"
    )


# ============================================================
# PRECISION-RECALL CURVE
# ============================================================

def save_precision_recall_curve(
    y_test,
    y_proba
):

    precision, recall, thresholds = (
        precision_recall_curve(
            y_test,
            y_proba
        )
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        recall,
        precision,
        lw=2
    )

    plt.xlabel(
        "Recall"
    )

    plt.ylabel(
        "Precision"
    )

    plt.title(
        "Precision-Recall Curve"
    )

    plt.grid(
        alpha=0.3
    )

    path = os.path.join(
        ARTIFACTS_DIR,
        "precision_recall_curve.png"
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"[ARTIFACT] Precision-Recall curve saved : "
        f"{path}"
    )


# ============================================================
# BEST THRESHOLDS
# ============================================================

def display_best_thresholds(
    results
):

    # --------------------------------------------------------
    # Best F1
    # --------------------------------------------------------

    best_f1 = results.loc[
        results["f1"].idxmax()
    ]

    # --------------------------------------------------------
    # Best recall with precision >= 30%
    # --------------------------------------------------------

    candidates_30 = results[
        results["precision"] >= 0.30
    ]

    if len(candidates_30) > 0:

        best_recall_30 = candidates_30.loc[
            candidates_30["recall"].idxmax()
        ]

    else:

        best_recall_30 = None

    # --------------------------------------------------------
    # Best recall with precision >= 40%
    # --------------------------------------------------------

    candidates_40 = results[
        results["precision"] >= 0.40
    ]

    if len(candidates_40) > 0:

        best_recall_40 = candidates_40.loc[
            candidates_40["recall"].idxmax()
        ]

    else:

        best_recall_40 = None

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "BEST THRESHOLDS"
    )

    print(
        "=" * 70
    )

    print(
        "\n🎯 Best F1:"
    )

    print(
        f"   Threshold : "
        f"{best_f1['threshold']:.2f}"
    )

    print(
        f"   Precision : "
        f"{best_f1['precision']:.3f}"
    )

    print(
        f"   Recall    : "
        f"{best_f1['recall']:.3f}"
    )

    print(
        f"   F1        : "
        f"{best_f1['f1']:.3f}"
    )

    if best_recall_30 is not None:

        print(
            "\n🎯 Best Recall "
            "(Precision >= 30%):"
        )

        print(
            f"   Threshold : "
            f"{best_recall_30['threshold']:.2f}"
        )

        print(
            f"   Precision : "
            f"{best_recall_30['precision']:.3f}"
        )

        print(
            f"   Recall    : "
            f"{best_recall_30['recall']:.3f}"
        )

    if best_recall_40 is not None:

        print(
            "\n🎯 Best Recall "
            "(Precision >= 40%):"
        )

        print(
            f"   Threshold : "
            f"{best_recall_40['threshold']:.2f}"
        )

        print(
            f"   Precision : "
            f"{best_recall_40['precision']:.3f}"
        )

        print(
            f"   Recall    : "
            f"{best_recall_40['recall']:.3f}"
        )

    print(
        "\n" + "=" * 70
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n" + "=" * 70
    )

    print(
        "FRAUD MODEL — THRESHOLD DIAGNOSTIC"
    )

    print(
        "=" * 70
    )

    print(
        "\n⚠️  No model training will be performed."
    )

    print(
        "This script only evaluates the existing model."
    )


    # ========================================================
    # STEP 1 — LOAD DATA
    # ========================================================

    print(
        "\n[STEP 1] Loading data..."
    )

    df = load_training_data()

    check_data_quality(
        df
    )


    # ========================================================
    # STEP 2 — PREPROCESS
    # ========================================================

    print(
        "\n[STEP 2] Preprocessing..."
    )

    X, y = preprocess(
        df
    )


    # ========================================================
    # STEP 3 — SAME SPLIT AS TRAIN.PY
    # ========================================================

    print(
        "\n[STEP 3] Reproducing train/test split..."
    )

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = train_test_split(

        X,
        y,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=y
    )


    print(
        f"[SPLIT] Train : "
        f"{len(X_train):,}"
    )

    print(
        f"[SPLIT] Test  : "
        f"{len(X_test):,}"
    )

    print(
        f"[SPLIT] Test fraud rate : "
        f"{y_test.mean() * 100:.2f}%"
    )


    # ========================================================
    # STEP 4 — LOAD EXISTING MODEL
    # ========================================================

    print(
        "\n[STEP 4] Loading existing model..."
    )

    model = load_model()


    # ========================================================
    # STEP 5 — PREDICT PROBABILITIES
    # ========================================================

    print(
        "\n[STEP 5] Generating probabilities..."
    )

    y_proba = model.predict_proba(
        X_test
    )[:, 1]

    print(
        "[PREDICTION] ✅ Probabilities generated"
    )


    # ========================================================
    # STEP 6 — GLOBAL METRICS
    # ========================================================

    auc_roc = roc_auc_score(
        y_test,
        y_proba
    )

    auc_pr = average_precision_score(
        y_test,
        y_proba
    )


    print(
        "\n" + "=" * 70
    )

    print(
        "GLOBAL MODEL METRICS"
    )

    print(
        "=" * 70
    )

    print(
        f"AUC-ROC : {auc_roc:.4f}"
    )

    print(
        f"AUC-PR  : {auc_pr:.4f}"
    )


    # ========================================================
    # STEP 7 — THRESHOLD ANALYSIS
    # ========================================================

    results = analyze_thresholds(
        y_test,
        y_proba
    )


    # ========================================================
    # STEP 8 — SAVE RESULTS
    # ========================================================

    save_results(
        results
    )


    # ========================================================
    # STEP 9 — BEST THRESHOLDS
    # ========================================================

    display_best_thresholds(
        results
    )


    # ========================================================
    # STEP 10 — CONFUSION MATRIX
    # ========================================================

    save_confusion_matrix(
        y_test,
        y_proba,
        threshold=0.50
    )


    # ========================================================
    # STEP 11 — PRECISION-RECALL CURVE
    # ========================================================

    save_precision_recall_curve(
        y_test,
        y_proba
    )


    # ========================================================
    # END
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "✅ DIAGNOSTIC COMPLETED"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()