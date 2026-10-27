import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load the original dataset
df = pd.read_excel("data/Telco_customer_churn.xlsx")

output_dir = Path("reports/figures")
output_dir.mkdir(parents=True, exist_ok=True)

# Prepare churn indicator
df["Churn Value"] = pd.to_numeric(df["Churn Value"])

# 1. Churn rate by contract
contract_churn = (
    df.groupby("Contract")["Churn Value"].mean().mul(100)
)

contract_churn.sort_values().plot(
    kind="bar",
    title="Churn Rate by Contract Type",
    ylabel="Churn rate (%)",
    xlabel="Contract type",
    rot=0,
)
plt.tight_layout()
plt.savefig(output_dir / "churn_by_contract.png", dpi=150)
plt.close()

# 2. Tenure distribution by churn status
for label, group in df.groupby("Churn Label"):
    plt.hist(
        group["Tenure Months"],
        bins=20,
        alpha=0.5,
        label=label,
    )

plt.title("Customer Tenure by Churn Status")
plt.xlabel("Tenure (months)")
plt.ylabel("Number of customers")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "tenure_by_churn.png", dpi=150)
plt.close()

# 3. Monthly charges by churn status
df.boxplot(column="Monthly Charges", by="Churn Label")
plt.title("Monthly Charges by Churn Status")
plt.suptitle("")
plt.xlabel("Churn status")
plt.ylabel("Monthly charges ($)")
plt.tight_layout()
plt.savefig(output_dir / "charges_by_churn.png", dpi=150)
plt.close()

print("EDA charts saved:")
for path in sorted(output_dir.glob("*.png")):
    print(path)

