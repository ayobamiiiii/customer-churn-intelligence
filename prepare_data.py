import pandas as pd
from pathlib import Path

# Load the original dataset
df = pd.read_excel("data/Telco_customer_churn.xlsx")

# Convert the target into a binary value
df["Churn Value"] = pd.to_numeric(df["Churn Value"], errors="raise")
df["Churn Label"] = df["Churn Label"].map({"No": 0, "Yes": 1})

# Remove columns that reveal the outcome or identify a customer
exclude_columns = [
    "CustomerID",
    "Churn Label",
    "Churn Value",
    "Churn Score",
    "Churn Reason",
]

X = df.drop(columns=exclude_columns, errors="ignore")
y = df["Churn Value"].astype(int)

# Exclude other potential post-outcome or derived fields for now
X = X.drop(columns=["CLTV"], errors="ignore")

# Save a baseline modelling dataset
output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

model_df = X.copy()
model_df["Churn Value"] = y
model_df.to_csv(output_dir / "churn_baseline.csv", index=False)

print("Baseline dataset created.")
print("Features:", X.shape[1])
print("Rows:", X.shape[0])
print("\nTarget distribution:")
print(y.value_counts())
print("\nRemaining columns:")
print(X.columns.tolist())
