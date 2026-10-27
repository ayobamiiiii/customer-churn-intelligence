
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load the tuned model
artifact = joblib.load("models/churn_model_tuned.joblib")
model = artifact["model"]
features = artifact["features"]

# Extract LightGBM feature importance
importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_,
})

importance = importance.sort_values(
    "Importance", ascending=False
)

print("\nTOP 15 FEATURES")
print(importance.head(15).to_string(index=False))

# Plot the top 15 features
top = importance.head(15).sort_values("Importance")

plt.figure(figsize=(10, 7))
plt.barh(top["Feature"], top["Importance"])
plt.title("Top 15 Features — Tuned Churn Model")
plt.xlabel("Split Importance")
plt.ylabel("Feature")
plt.tight_layout()

Path("reports/figures").mkdir(
    parents=True, exist_ok=True
)

plt.savefig(
    "reports/figures/tuned_feature_importance.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()
print("\nChart saved to reports/figures/tuned_feature_importance.png")

