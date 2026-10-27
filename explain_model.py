
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# Load the saved model
saved = joblib.load("models/churn_model_tuned.joblib")
model = saved["model"]
features = saved["features"]

# Load the processed dataset
df = pd.read_csv("data/processed/churn_baseline.csv")

# Convert charges to numeric
df["Total Charges"] = pd.to_numeric(
    df["Total Charges"], errors="coerce"
)

# Remove identifiers and leakage-prone fields
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

# Keep rows without missing feature values
valid = X.notna().all(axis=1)
X = X.loc[valid]

# Match the model's training features
X = pd.get_dummies(X, drop_first=True)
X = X.reindex(columns=features, fill_value=0)

# Explain a sample to keep computation manageable
sample = X.sample(
    n=min(500, len(X)),
    random_state=42
)

# Calculate SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer(sample)

# Create output directory
Path("reports/figures").mkdir(
    parents=True, exist_ok=True
)

# Global feature importance
shap.plots.bar(shap_values, max_display=15, show=False)
plt.tight_layout()
plt.savefig(
    "reports/figures/shap_feature_importance.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close()

# Show how features affect churn predictions
shap.plots.beeswarm(
    shap_values, max_display=15, show=False
)
plt.tight_layout()
plt.savefig(
    "reports/figures/shap_summary.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close()

print("SHAP analysis completed.")
print("Saved:")
print("reports/figures/shap_feature_importance.png")
print("reports/figures/shap_summary.png")

