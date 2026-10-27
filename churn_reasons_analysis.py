import pandas as pd
import matplotlib.pyplot as plt

# Load the original dataset
df = pd.read_excel("data/Telco_customer_churn.xlsx")

# Keep customers who churned and have a recorded reason
churned = df[df["Churn Value"] == 1].copy()

reasons = churned["Churn Reason"].dropna()
reasons = reasons[reasons.str.strip() != ""]

# Count and rank the top 10 reasons
top_reasons = reasons.value_counts().head(10).sort_values()

print("\nTOP 10 CHURN REASONS")
print(top_reasons.sort_values(ascending=False))

# Create horizontal bar chart
plt.figure(figsize=(11, 7))
top_reasons.plot(kind="barh", color="steelblue")

plt.title("Top 10 Recorded Customer Churn Reasons")
plt.xlabel("Number of Customers")
plt.ylabel("Churn Reason")
plt.tight_layout()

plt.savefig(
    "reports/figures/top_churn_reasons.png",
    dpi=300,
    bbox_inches="tight"
)

print("\nChart saved to reports/figures/top_churn_reasons.png")
plt.show()
