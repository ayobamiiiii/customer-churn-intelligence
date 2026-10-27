
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

# 1. Load the baseline dataset
df = pd.read_csv("data/processed/churn_baseline.csv")

# 2. Prepare the required columns
df["Tenure Months"] = pd.to_numeric(
    df["Tenure Months"], errors="coerce"
)
df["Churn Value"] = pd.to_numeric(
    df["Churn Value"], errors="coerce"
)

df = df.dropna(subset=["Tenure Months", "Churn Value"])
df = df[df["Tenure Months"] >= 0].copy()

# 3. Define churn events
# 1 = churned, 0 = still active
df["event"] = df["Churn Value"].astype(int)

# 4. Fit Kaplan-Meier survival model
kmf = KaplanMeierFitter()

kmf.fit(
    durations=df["Tenure Months"],
    event_observed=df["event"],
    label="Customer survival"
)

# 5. Display results
print("Total customers:", len(df))
print("Churn events:", df["event"].sum())
print("Non-churn events:", (df["event"] == 0).sum())
median_time = kmf.median_survival_time_

if median_time == float("inf"):
    print("Median survival time: Not reached")
else:
    print("Median survival time:", median_time, "months")

# 6. Plot the survival curve
plt.figure(figsize=(10, 6))
kmf.plot_survival_function()

plt.title("Kaplan-Meier Customer Survival Curve")
plt.xlabel("Tenure (months)")
plt.ylabel("Probability of Remaining Unchurned")
plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 7. Save the chart
plt.savefig(
    "reports/figures/kaplan_meier_survival.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Chart saved successfully.")




# 8. Compare survival by contract type
print("\nSURVIVAL BY CONTRACT TYPE")

plt.figure(figsize=(10, 6))

for contract in ["Month-to-month", "One year", "Two year"]:
    group = df[df["Contract"] == contract]

    if group.empty:
        print(f"No data found for contract: {contract}")
        continue

    kmf_contract = KaplanMeierFitter()
    kmf_contract.fit(
        durations=group["Tenure Months"],
        event_observed=group["event"],
        label=contract
    )

    kmf_contract.plot_survival_function()

    print(
        f"{contract}: "
        f"{len(group)} customers, "
        f"median survival = {kmf_contract.median_survival_time_}"
    )

plt.title("Customer Survival by Contract Type")
plt.xlabel("Tenure (months)")
plt.ylabel("Probability of Remaining Unchurned")
plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    "reports/figures/survival_by_contract.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Saved: reports/figures/survival_by_contract.png")

