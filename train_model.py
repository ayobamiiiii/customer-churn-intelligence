
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

# 1. Load the processed dataset
df = pd.read_csv("data/processed/churn_baseline.csv")

# 2. Convert Total Charges to numeric
df["Total Charges"] = pd.to_numeric(
    df["Total Charges"], errors="coerce"
)

# 3. Remove identifiers and fields that could leak the answer
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
y = df["Churn Value"].astype(int)

# 4. Remove rows with missing values
valid = X.notna().all(axis=1) & y.notna()
X = X.loc[valid]
y = y.loc[valid]

# 5. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# 6. Encode categorical features
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# Ensure training and testing have identical columns
X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=0,
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Features after encoding:", X_train.shape[1])

# 7. Train the LightGBM classifier
model = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    class_weight="balanced",
    random_state=42,
    verbosity=-1,
)

model.fit(X_train, y_train)

# 8. Predict churn probabilities
y_prob = model.predict_proba(X_test)[:, 1]

# 9. Compare different classification thresholds
print("\nTHRESHOLD COMPARISON")
print("-" * 65)

for threshold in [0.30, 0.35, 0.40, 0.45, 0.50]:
    y_pred_temp = (y_prob >= threshold).astype(int)

    precision = precision_score(
        y_test, y_pred_temp, zero_division=0
    )
    recall = recall_score(y_test, y_pred_temp)
    f1 = f1_score(y_test, y_pred_temp, zero_division=0)

    print(
        f"Threshold: {threshold:.2f} | "
        f"Precision: {precision:.3f} | "
        f"Recall: {recall:.3f} | "
        f"F1: {f1:.3f}"
    )

# 10. Evaluate using the initial threshold of 0.35
threshold = 0.35
y_pred = (y_prob >= threshold).astype(int)

print(f"\nFINAL RESULTS AT THRESHOLD {threshold:.2f}")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(
    precision_score(y_test, y_pred, zero_division=0), 4
))
print("Recall:", round(recall_score(y_test, y_pred), 4))
print("F1 score:", round(
    f1_score(y_test, y_pred, zero_division=0), 4
))
print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report:")
print(classification_report(
    y_test, y_pred, zero_division=0
))

# 11. Save the trained model and its feature columns
Path("models").mkdir(exist_ok=True)

joblib.dump(
    {
        "model": model,
        "features": X_train.columns.tolist(),
        "threshold": threshold,
    },
    "models/churn_model.joblib",
)

print("\nModel saved to models/churn_model.joblib")

