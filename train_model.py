
import pandas as pd
import lightgbm as lgb
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# 1. Load data
df = pd.read_csv("data/processed/churn_baseline.csv")

df["Total Charges"] = pd.to_numeric(
    df["Total Charges"], errors="coerce"
)

# 2. Separate target and remove leakage/identifier columns
drop_cols = [
    "Churn Value",
    "Count",
    "Country",
    "State",
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude",
    "Churn Score",
    "CLTV",
    "Churn Label",
    "Churn Reason",
]

X = df.drop(columns=drop_cols, errors="ignore")
y = pd.to_numeric(df["Churn Value"], errors="coerce")

# 3. Clean missing values
valid = X.notna().all(axis=1) & y.notna()
X = X.loc[valid].copy()
y = y.loc[valid].astype(int)

# 4. Encode categorical variables consistently
X = pd.get_dummies(X, drop_first=True, dtype=int)

# 5. Split into train (60%), validation (20%), test (20%)
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval,
    y_trainval,
    test_size=0.25,
    random_state=42,
    stratify=y_trainval,
)

print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Testing rows:", len(X_test))
print("Features:", X_train.shape[1])

# 6. Train LightGBM
model = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    class_weight="balanced",
    random_state=42,
    verbosity=-1,
)

model.fit(X_train, y_train)

# 7. Select threshold using validation data only
val_prob = model.predict_proba(X_val)[:, 1]

thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
results = []

print("\nVALIDATION THRESHOLD COMPARISON")
print("-" * 65)

for threshold in thresholds:
    val_pred = (val_prob >= threshold).astype(int)

    precision = precision_score(
        y_val, val_pred, zero_division=0
    )
    recall = recall_score(
        y_val, val_pred, zero_division=0
    )
    f1 = f1_score(
        y_val, val_pred, zero_division=0
    )

    results.append((f1, threshold))

    print(
        f"Threshold: {threshold:.2f} | "
        f"Precision: {precision:.3f} | "
        f"Recall: {recall:.3f} | "
        f"F1: {f1:.3f}"
    )

# Maximize validation F1; prefer the higher threshold in a tie
_, best_threshold = max(results, key=lambda item: (item[0], item[1]))

# 8. Evaluate once on the untouched test set
test_prob = model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= best_threshold).astype(int)

print(f"\nFINAL TEST RESULTS — THRESHOLD {best_threshold:.2f}")
print("Accuracy:", round(accuracy_score(y_test, test_pred), 4))
print("Precision:", round(
    precision_score(y_test, test_pred, zero_division=0), 4
))
print("Recall:", round(
    recall_score(y_test, test_pred, zero_division=0), 4
))
print("F1 score:", round(
    f1_score(y_test, test_pred, zero_division=0), 4
))
print("ROC-AUC:", round(roc_auc_score(y_test, test_prob), 4))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, test_pred))

print("\nClassification report:")
print(classification_report(
    y_test, test_pred, zero_division=0
))

# 9. Save model and feature metadata
Path("models").mkdir(exist_ok=True)

joblib.dump(
    {
        "model": model,
        "features": X_train.columns.tolist(),
        "threshold": best_threshold,
    },
    "models/churn_model.joblib",
)

print("\nModel saved to models/churn_model.joblib")

