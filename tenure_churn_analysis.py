import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/processed/churn_baseline.csv")

# Create tenure groups
bins = [0, 12, 24, 36, 48, 60, 72]
labels = [
    "1-12 months",
    "13-24 months",
    "25-36 months",
    "37-48 months",
    "49-60 months",
    "61-72 months"
]

df["Tenure Group"] = pd.cut(
    df["Tenure Months"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

# Calculate churn rate by tenure group
summary = df.groupby(
    "Tenure Group", observed=False
).agg(
    Customers=("Churn Value", "count"),
    Churned=("Churn Value", "sum"),
    Churn_Rate=("Churn Value", "mean")
)

summary["Churn_Rate"] = (
    summary["Churn_Rate"] * 100
).round(2)

print("\nCHURN RATE BY TENURE")
print(summary)

# Plot results
plt.figure(figsize=(10, 6))
summary["Churn_Rate"].plot(
    kind="bar",
    color="steelblue"
)

plt.title("Customer Churn Rate by Tenure")
plt.xlabel("Customer Tenure")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(
    "reports/figures/churn_by_tenure_group.png",
    dpi=300,
    bbox_inches="tight"
)

print("\nChart saved successfully.")
plt.show()
