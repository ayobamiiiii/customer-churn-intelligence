
import pandas as pd
import lightgbm as lgb
import joblib

from pathlib import Path
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
)
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
)

# 1. Load and prepare the dataset
df = pd.read_csv("data/processed/churn_baseline.csv")

df["Total Charges"] = pd.to_numeric(
    df["Total Charges"], errors="coerce"
)

drop_cols = [
    "Churn Value", "Count", "Country", "State",
    "City", "Zip Code", "Lat Long", "Latitude",
    "Longitude", "Churn Score", "CLTV",
    "Churn Label", "Churn Reason",
]

X = df.drop(columns=drop_cols, errors="ignore")
y = pd.to_numeric(df["Churn Value"], errors="coerce")

valid = X.notna().all(axis=1) & y.notna()
X = X.loc[valid].copy()
y = y.loc[valid].astype(int)

# Encode before splitting to keep a consistent feature space
X = pd.get_dummies(X, drop_first=True, dtype=int)

# 2. Create train, validation and test sets
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval,
    test_size=0.25,
    random_state=42,
    stratify=y_trainval,
)

print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Testing:", len(X_test))
print("Features:", X_train.shape[1])

# 3. Define hyperparameter search
parameter_space = {
    "n_estimators": [200, 300, 500, 700],
    "learning_rate": [0.01, 0.03, 0.05, 0.08],
    "num_leaves": [15, 20, 31, 40],
    "max_depth": [-1, 5, 8, 12],
    "min_child_samples": [10, 20, 30, 50],
    "reg_alpha": [0, 0.1, 0.5, 1.0],
    "reg_lambda": [0, 0.1, 0.5, 1.0],
    "colsample_bytree": [0.8, 0.9, 1.0],
}

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42,
)

search = RandomizedSearchCV(
    estimator=lgb.LGBMClassifier(
        class_weight="balanced",
        random_state=42,
        verbosity=-1,
        n_jobs=1,
    ),
    param_distributions=parameter_space,
    n_iter=20,
    scoring="roc_auc",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=1,
    refit=True,
)

# 4. Fit using training data only
search.fit(X_train, y_train)

best_model = search.best_estimator_

print("\nBEST PARAMETERS")
print(search.best_params_)
print("Best cross-validation ROC-AUC:", round(search.best_score_, 4))

# 5. Choose threshold on validation data only
val_prob = best_model.predict_proba(X_val)[:, 1]

threshold_results = []

print("\nVALIDATION THRESHOLDS")
for threshold in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
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

    threshold_results.append((f1, threshold))

    print(
        f"Threshold: {threshold:.2f} | "
        f"Precision: {precision:.3f} | "
        f"Recall: {recall:.3f} | F1: {f1:.3f}"
    )

_, best_threshold = max(
    threshold_results, key=lambda result: (result[0], result[1])
)

# 6. Evaluate once on the held-out test set
test_prob = best_model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= best_threshold).astype(int)

print(f"\nTUNED TEST RESULTS — THRESHOLD {best_threshold:.2f}")
print("Accuracy:", round(accuracy_score(y_test, test_pred), 4))
print("Precision:", round(
    precision_score(y_test, test_pred, zero_division=0), 4
))
print("Recall:", round(
    recall_score(y_test, test_pred, zero_division=0), 4
))
print("F1:", round(f1_score(y_test, test_pred, zero_division=0), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, test_prob), 4))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, test_pred))

print("\nClassification report:")
print(classification_report(
    y_test, test_pred, zero_division=0
))

# 7. Save tuned model separately to preserve baseline
Path("models").mkdir(exist_ok=True)

joblib.dump(
    {
        "model": best_model,
        "features": X_train.columns.tolist(),
        "threshold": best_threshold,
    },
    "models/churn_model_tuned.joblib",
)

print("\nTuned model saved to models/churn_model_tuned.joblib")


