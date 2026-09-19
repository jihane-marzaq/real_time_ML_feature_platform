import json
import os

from data_loader import load_training_data
from train import preprocess
from config import FEATURE_COLS, TARGET_COL


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ARTIFACTS_DIR = os.path.join(
    BASE_DIR,
    "artifacts"
)

METADATA_PATH = os.path.join(
    ARTIFACTS_DIR,
    "preprocessing_metadata.json"
)

os.makedirs(
    ARTIFACTS_DIR,
    exist_ok=True
)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("[METADATA] Loading training data...")

df = load_training_data()

print(f"[METADATA] Rows: {len(df):,}")
print(
    f"[METADATA] Fraud rate: "
    f"{df[TARGET_COL].mean() * 100:.2f}%"
)


# ============================================================
# 2. IDENTIFY CATEGORICAL COLUMNS
# ============================================================

X_original = df[FEATURE_COLS].copy()

categorical_cols = (
    X_original
    .select_dtypes(include=["object", "string"])
    .columns
    .tolist()
)

numeric_cols = [
    col
    for col in FEATURE_COLS
    if col not in categorical_cols
]

print("\n[METADATA] Categorical columns:")
for col in categorical_cols:
    print(f"  - {col}")

print("\n[METADATA] Numeric columns:")
for col in numeric_cols:
    print(f"  - {col}")


# ============================================================
# 3. RUN THE EXACT TRAINING PREPROCESSING
# ============================================================

print("\n[METADATA] Running training preprocessing...")

X_processed, y, encoders = preprocess(df)

print(
    f"[METADATA] Processed shape: "
    f"{X_processed.shape}"
)


# ============================================================
# 4. NUMERIC MEDIANS
# ============================================================

numeric_medians = {}

for col in numeric_cols:

    median_value = X_processed[col].median()

    numeric_medians[col] = float(median_value)

    print(
        f"[MEDIAN] {col}: "
        f"{median_value}"
    )


# ============================================================
# 5. SAVE METADATA
# ============================================================

# Convertir les valeurs numpy.int64 des mappings
# en int Python standard pour JSON
json_safe_encoders = {
    col: {
        str(category): int(encoded_value)
        for category, encoded_value in mapping.items()
    }
    for col, mapping in encoders.items()
}


metadata = {
    "model_version": 5,
    "feature_cols": FEATURE_COLS,
    "categorical_mappings": json_safe_encoders,
    "numeric_medians": numeric_medians
}


with open(
    METADATA_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
        ensure_ascii=False
    )


print("\n[METADATA] ✅ Saved:")
print(METADATA_PATH)