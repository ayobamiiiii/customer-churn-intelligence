
import os
import glob
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

# PAGE CONFIGURATION — MUST BE THE FIRST STREAMLIT COMMAND

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# PROJECT PATHS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR, "models", "churn_model_tuned.joblib"
)

# Exact 30 features expected by the saved model
FEATURES = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
    "Gender_Male",
    "Senior Citizen_Yes",
    "Partner_Yes",
    "Dependents_Yes",
    "Phone Service_Yes",
    "Multiple Lines_No phone service",
    "Multiple Lines_Yes",
    "Internet Service_Fiber optic",
    "Internet Service_No",
    "Online Security_No internet service",
    "Online Security_Yes",
    "Online Backup_No internet service",
    "Online Backup_Yes",
    "Device Protection_No internet service",
    "Device Protection_Yes",
    "Tech Support_No internet service",
    "Tech Support_Yes",
    "Streaming TV_No internet service",
    "Streaming TV_Yes",
    "Streaming Movies_No internet service",
    "Streaming Movies_Yes",
    "Contract_One year",
    "Contract_Two year",
    "Paperless Billing_Yes",
    "Payment Method_Credit card (automatic)",
    "Payment Method_Electronic check",
    "Payment Method_Mailed check",
]

@st.cache_data
def load_customer_data():
    """Find and load the customer churn dataset."""

    preferred_paths = [
        os.path.join(BASE_DIR, "data", "Telco_customer_churn.xlsx"),
        os.path.join(BASE_DIR, "Telco_customer_churn.xlsx"),
        os.path.join(BASE_DIR, "data", "telco_customer_churn.csv"),
        os.path.join(BASE_DIR, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv"),
    ]

    candidates = preferred_paths.copy()

    for pattern in [
        os.path.join(BASE_DIR, "**", "*.csv"),
        os.path.join(BASE_DIR, "**", "*.xlsx"),
    ]:
        candidates.extend(glob.glob(pattern, recursive=True))

    checked = set()

    for path in candidates:
        if path in checked or not os.path.isfile(path):
            continue

        checked.add(path)

        try:
            if path.lower().endswith(".xlsx"):
                data = pd.read_excel(path)
            else:
                data = pd.read_csv(path)

            if not data.empty:
                return data

        except Exception as exc:
            st.warning(f"Could not load {os.path.basename(path)}: {exc}")

    return None

@st.cache_resource
def load_saved_model():
    """Load the saved model artifact."""

    if not os.path.isfile(MODEL_PATH):
        return None, (
            f"Model file not found: {MODEL_PATH}"
        )

    try:
        artifact = joblib.load(MODEL_PATH)

        # Your saved artifact is a dictionary containing
        # model, features, and threshold.
        if isinstance(artifact, dict):
            model = artifact.get("model")
            features = artifact.get("features", FEATURES)
            threshold = float(artifact.get("threshold", 0.5))
        else:
            model = artifact
            features = FEATURES
            threshold = 0.5

        if model is None:
            return None, "The saved model artifact has no model."

        if list(features) != FEATURES:
            return None, (
                "The feature list in the saved model does not match "
                "the expected 30 features. Use the same feature "
                "order and encoding as during training."
            )

        return {
            "model": model,
            "features": list(features),
            "threshold": threshold,
        }, None

    except Exception as exc:
        return None, f"Could not load the model: {exc}"


# DATA CLEANING AND STANDARDIZATION

def standardize_data(data):
    """Normalize common churn dataset column formats."""

    df = data.copy()

    # Normalize binary churn labels.
    if "Churn Value" not in df.columns:
        if "Churn Label" in df.columns:
            df["Churn Value"] = (
                df["Churn Label"]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"yes": 1, "no": 0, "1": 1, "0": 0})
            )
        elif "Churn" in df.columns:
            df["Churn Value"] = (
                df["Churn"]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"yes": 1, "no": 0, "1": 1, "0": 0})
            )

    # Standardize numerical columns.
    for column in [
        "Tenure Months",
        "Monthly Charges",
        "Total Charges",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column], errors="coerce"
            )

    if "Churn Value" in df.columns:
        df["Churn Value"] = pd.to_numeric(
            df["Churn Value"], errors="coerce"
        )

    if "Tenure Months" in df.columns:
        df["Tenure Months"] = df["Tenure Months"].fillna(0)

    if "Monthly Charges" in df.columns:
        df["Monthly Charges"] = df["Monthly Charges"].fillna(
            df["Monthly Charges"].median()
        )

    if "Total Charges" in df.columns:
        df["Total Charges"] = df["Total Charges"].fillna(0)

    return df


def calculate_churn_rate(frame):
    """Calculate churn as a percentage."""

    if frame.empty or "Churn Value" not in frame.columns:
        return 0.0

    values = pd.to_numeric(
        frame["Churn Value"], errors="coerce"
    ).dropna()

    if values.empty:
        return 0.0

    return float(values.mean() * 100)


def get_churn_column(df):
    """Return the standardized churn column if available."""

    return "Churn Value" if "Churn Value" in df.columns else None


# CHARGE CONSISTENCY VALIDATION

def check_charge_consistency(tenure, monthly_charges, total_charges):
    """
    Compare cumulative charges with tenure multiplied by
    monthly charges. This is a plausibility check, not an
    exact accounting reconciliation.
    """

    expected = tenure * monthly_charges
    difference = abs(total_charges - expected)

    # Allow for changes in plans, taxes, discounts, and billing.
    tolerance = max(50.0, expected * 0.15)

    if tenure == 0:
        consistent = total_charges <= 50.0
    else:
        consistent = difference <= tolerance

    return {
        "expected": expected,
        "difference": difference,
        "tolerance": tolerance,
        "consistent": consistent,
    }


# BUILD EXACT MODEL INPUT

def build_prediction_row(values):
    """Build a single row with the model's exact 30 features."""

    row = {feature: 0 for feature in FEATURES}

    row["Tenure Months"] = float(values["tenure"])
    row["Monthly Charges"] = float(values["monthly"])
    row["Total Charges"] = float(values["total"])

    # Binary features
    row["Gender_Male"] = int(values["gender"] == "Male")
    row["Senior Citizen_Yes"] = int(values["senior"] == "Yes")
    row["Partner_Yes"] = int(values["partner"] == "Yes")
    row["Dependents_Yes"] = int(values["dependents"] == "Yes")
    row["Phone Service_Yes"] = int(values["phone"] == "Yes")

    if values["phone"] == "No":
        row["Multiple Lines_No phone service"] = 1
    elif values["multiple_lines"] == "Yes":
        row["Multiple Lines_Yes"] = 1

    if values["internet"] == "Fiber optic":
        row["Internet Service_Fiber optic"] = 1
    elif values["internet"] == "No":
        row["Internet Service_No"] = 1

    services = [
        ("security", "Online Security"),
        ("backup", "Online Backup"),
        ("protection", "Device Protection"),
        ("support", "Tech Support"),
        ("tv", "Streaming TV"),
        ("movies", "Streaming Movies"),
    ]

    for key, feature_prefix in services:
        choice = values[key]

        if values["internet"] == "No":
            row[f"{feature_prefix}_No internet service"] = 1
        elif choice == "Yes":
            row[f"{feature_prefix}_Yes"] = 1

    if values["contract"] == "One year":
        row["Contract_One year"] = 1
    elif values["contract"] == "Two year":
        row["Contract_Two year"] = 1

    row["Paperless Billing_Yes"] = int(
        values["paperless"] == "Yes"
    )

    if values["payment"] == "Credit card (automatic)":
        row["Payment Method_Credit card (automatic)"] = 1
    elif values["payment"] == "Electronic check":
        row["Payment Method_Electronic check"] = 1
    elif values["payment"] == "Mailed check":
        row["Payment Method_Mailed check"] = 1

    return pd.DataFrame([row], columns=FEATURES)


# LOAD RESOURCES

raw_df = load_customer_data()

if raw_df is None:
    st.error(
        "Customer dataset not found. Put your churn CSV in the "
        "project's data/ folder or project root."
    )
    st.stop()

df = standardize_data(raw_df)

if "Churn Value" not in df.columns:
    st.error(
        "The dataset has no usable churn target column. "
        "Expected Churn Value, Churn Label, or Churn."
    )
    st.stop()

model_bundle, model_error = load_saved_model()

# APP HEADER

st.title("Customer Churn Intelligence")
st.caption(
    "Telecommunications analytics | Churn prediction | "
    "Model explainability | Customer retention"
)
st.write(
    "Identify customer churn patterns, estimate churn risk, "
    "and support data-driven retention decisions."
)

# NAVIGATION

with st.sidebar:
    st.title("Navigation")

    page = st.radio(
        "Select a page",
        ["Dashboard", "Predict Churn", "Model Insights"],
        key="page_navigation",
    )

    st.divider()
    st.caption("Customer Churn Intelligence")
    st.caption("Built with Python and Streamlit")

# PAGE 1: DASHBOARD

if page == "Dashboard":

    st.header("Customer Churn Dashboard")

    total_customers = len(df)
    churned = int((df["Churn Value"] == 1).sum())
    retained = total_customers - churned
    churn_rate = (
        churned / total_customers * 100
        if total_customers else 0
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Customers", f"{total_customers:,}")
    c2.metric("Churned Customers", f"{churned:,}")
    c3.metric("Retained Customers", f"{retained:,}")
    c4.metric("Churn Rate", f"{churn_rate:.1f}%")

    st.divider()

    # Churn by contract
    left, right = st.columns(2)

    with left:
        st.subheader("Churn Rate by Contract Type")

        if "Contract" in df.columns:
            contract_rates = (
                df.groupby("Contract", observed=False)["Churn Value"]
                .mean()
                .mul(100)
                .rename("Churn Rate (%)")
                .reset_index()
                .sort_values("Churn Rate (%)", ascending=False)
            )

            st.bar_chart(
                contract_rates,
                x="Contract",
                y="Churn Rate (%)",
            )
        else:
            st.info("Contract column is unavailable.")

    # Churn by tenure
    with right:
        st.subheader("Churn Rate by Customer Tenure")

        if "Tenure Months" in df.columns:
            tenure_df = df[["Tenure Months", "Churn Value"]].copy()

            tenure_df["Tenure Group"] = pd.cut(
                tenure_df["Tenure Months"],
                bins=[-1, 12, 24, 48, 60, 72],
                labels=[
                    "0–12 months",
                    "13–24 months",
                    "25–48 months",
                    "49–60 months",
                    "61–72 months",
                ],
                include_lowest=True,
            )

            tenure_rates = (
                tenure_df.groupby(
                    "Tenure Group",
                    observed=False,
                )["Churn Value"]
                .mean()
                .mul(100)
                .rename("Churn Rate (%)")
                .reset_index()
                .dropna()
            )

            st.bar_chart(
                tenure_rates,
                x="Tenure Group",
                y="Churn Rate (%)",
            )
        else:
            st.info("Tenure column is unavailable.")

    st.divider()

    # Churn by monthly charges
    st.subheader("Churn Rate by Monthly Charges")

    if "Monthly Charges" in df.columns:
        charges_df = df[
            ["Monthly Charges", "Churn Value"]
        ].copy()

        charges_df["Monthly Charges"] = pd.to_numeric(
            charges_df["Monthly Charges"], errors="coerce"
        )

        charges_df = charges_df.dropna(
            subset=["Monthly Charges", "Churn Value"]
        )

        if not charges_df.empty:
            # Use string labels to avoid chart schema errors
            # caused by Interval objects from pd.cut().
            charges_df["Charge Band"] = pd.cut(
                charges_df["Monthly Charges"],
                bins=5,
                duplicates="drop",
            )

            charges_df = charges_df.dropna(
                subset=["Charge Band"]
            )

            charges_df["Charge Band"] = (
                charges_df["Charge Band"]
                .map(lambda interval: str(interval))
            )

            charge_rates = (
                charges_df.groupby(
                    "Charge Band", observed=False
                )["Churn Value"]
                .mean()
                .mul(100)
                .rename("Churn Rate (%)")
                .reset_index()
            )

            st.bar_chart(
                charge_rates,
                x="Charge Band",
                y="Churn Rate (%)",
            )
        else:
            st.info("No valid monthly charge values are available.")
    else:
        st.info("Monthly Charges column is unavailable.")

    st.divider()

    # Customer data preview
    st.subheader("Customer Data Preview")

    filter_option = st.selectbox(
        "Filter preview by",
        [
            "All customers",
            "Churned customers",
            "Retained customers",
        ],
    )

    preview = df.copy()

    if filter_option == "Churned customers":
        preview = preview[preview["Churn Value"] == 1]
    elif filter_option == "Retained customers":
        preview = preview[preview["Churn Value"] == 0]

    st.caption(
        f"Showing {min(10, len(preview)):,} of "
        f"{len(preview):,} matching customers."
    )

    # Compatible with Streamlit versions that do not accept
    # use_container_width=True.
    st.dataframe(
        preview.head(10),
        width="stretch",
    )


# PAGE 2: PREDICT CHURN

elif page == "Predict Churn":

    st.header("Customer Churn Risk Prediction")

    st.write(
        "Enter customer details to estimate churn probability "
        "using your saved, tuned LightGBM classifier."
    )

    if model_bundle is None:
        st.error(model_error)
        st.stop()

    model = model_bundle["model"]
    threshold = model_bundle["threshold"]

    st.caption(f"Decision threshold: {threshold:.2f}")

    with st.form("churn_prediction_form"):

        st.subheader("Customer and Billing Details")

        col1, col2, col3 = st.columns(3)

        with col1:
            tenure = st.number_input(
                "Tenure (months)",
                min_value=0,
                max_value=72,
                value=12,
                step=1,
            )

            monthly = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                max_value=2000.0,
                value=70.0,
                step=1.0,
            )

            total = st.number_input(
                "Total Charges",
                min_value=0.0,
                max_value=100000.0,
                value=840.0,
                step=10.0,
            )

        with col2:
            gender = st.selectbox(
                "Gender", ["Female", "Male"]
            )

            senior = st.selectbox(
                "Senior Citizen", ["No", "Yes"]
            )

            partner = st.selectbox(
                "Has Partner", ["No", "Yes"]
            )

            dependents = st.selectbox(
                "Has Dependents", ["No", "Yes"]
            )

        with col3:
            phone = st.selectbox(
                "Phone Service", ["Yes", "No"]
            )

            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["No", "Yes"] if phone == "Yes"
                else ["No phone service"],
            )

            internet = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"],
            )

            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"],
            )

        st.subheader("Additional Services")

        svc1, svc2, svc3 = st.columns(3)

        with svc1:
            security = st.selectbox(
                "Online Security",
                ["No", "Yes"],
                disabled=(internet == "No"),
            )

            backup = st.selectbox(
                "Online Backup",
                ["No", "Yes"],
                disabled=(internet == "No"),
            )

        with svc2:
            protection = st.selectbox(
                "Device Protection",
                ["No", "Yes"],
                disabled=(internet == "No"),
            )

            support = st.selectbox(
                "Tech Support",
                ["No", "Yes"],
                disabled=(internet == "No"),
            )

        with svc3:
            tv = st.selectbox(
                "Streaming TV",
                ["No", "Yes"],
                disabled=(internet == "No"),
            )

            movies = st.selectbox(
                "Streaming Movies",
                ["No", "Yes"],
                disabled=(internet == "No"),
            )

        paperless = st.selectbox(
            "Paperless Billing", ["No", "Yes"]
        )

        payment = st.selectbox(
            "Payment Method",
            [
                "Bank transfer (automatic)",
                "Credit card (automatic)",
                "Electronic check",
                "Mailed check",
            ],
        )

        submitted = st.form_submit_button(
            "Predict Churn Risk",
            use_container_width=True,
        )

    # Check billing consistency without pretending the
    # cumulative charges must match exactly.
    if submitted:

        charge_check = check_charge_consistency(
            float(tenure),
            float(monthly),
            float(total),
        )

        st.subheader("Charge Consistency Check")

        st.write(
            f"Estimated cumulative charges: "
            f"${charge_check['expected']:,.2f}"
        )

        st.write(
            f"Entered total charges: ${total:,.2f}"
        )

        st.write(
            f"Absolute difference: "
            f"${charge_check['difference']:,.2f}"
        )

        if charge_check["consistent"]:
            st.success(
                "Charges are within the configured tolerance. "
                "This is a plausibility check, not proof that "
                "the billing records are correct."
            )
        else:
            st.warning(
                "Charges differ materially from tenure × monthly "
                "charges. Verify the billing history before relying "
                "on this prediction. Plan changes, discounts, and "
                "billing adjustments may explain the difference."
            )

        values = {
            "tenure": tenure,
            "monthly": monthly,
            "total": total,
            "gender": gender,
            "senior": senior,
            "partner": partner,
            "dependents": dependents,
            "phone": phone,
            "multiple_lines": multiple_lines,
            "internet": internet,
            "contract": contract,
            "security": security,
            "backup": backup,
            "protection": protection,
            "support": support,
            "tv": tv,
            "movies": movies,
            "paperless": paperless,
            "payment": payment,
        }

        try:
            input_df = build_prediction_row(values)

            # Pass features in exactly the saved training order.
            input_df = input_df[model_bundle["features"]]

            probability = float(
                model.predict_proba(input_df)[0][1]
            )

            prediction = int(probability >= threshold)

            st.divider()
            st.subheader("Prediction Result")

            result1, result2 = st.columns(2)

            result1.metric(
                "Estimated Churn Probability",
                f"{probability * 100:.1f}%",
            )

            result2.metric(
                "Decision Threshold",
                f"{threshold * 100:.1f}%",
            )

            st.progress(
                min(max(probability, 0.0), 1.0)
            )

            if prediction == 1:
                st.error(
                    "HIGHER CHURN RISK: the model classifies "
                    "this customer as likely to churn."
                )
                st.write(
                    "Suggested action: review service issues, "
                    "contract options, billing concerns, and "
                    "retention offers."
                )
            else:
                st.success(
                    "LOWER CHURN RISK: the model classifies "
                    "this customer as likely to be retained."
                )
                st.write(
                    "Suggested action: maintain service quality "
                    "and continue monitoring customer engagement."
                )

            result_df = input_df.copy()
            result_df["Churn Probability"] = probability
            result_df["Predicted Churn"] = prediction

            st.download_button(
                "Download Prediction CSV",
                data=result_df.to_csv(index=False).encode("utf-8"),
                file_name="churn_prediction.csv",
                mime="text/csv",
            )

        except Exception as exc:
            st.error(
                f"Prediction failed: {exc}"
            )


# PAGE 3: MODEL INSIGHTS

elif page == "Model Insights":

    st.header("Model Evaluation")

    st.write(
        "The latest documented tuned-model test results:"
    )

    metrics = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-score",
            "ROC-AUC",
        ],
        "Score": [
            "74.13%",
            "50.85%",
            "79.68%",
            "62.08%",
            "84.97%",
        ],
    })

    st.dataframe(
        metrics,
        hide_index=True,
        width="stretch",
    )

    st.caption(
        "These are previously recorded evaluation results. "
        "They are not recalculated by this dashboard."
    )

    st.divider()
    st.subheader("Feature Importance")

    if model_bundle is None:
        st.warning(
            "The saved model could not be loaded, so feature "
            "importance is unavailable."
        )
    else:
        model = model_bundle["model"]

        if hasattr(model, "feature_importances_"):
            importance = pd.DataFrame({
                "Feature": model_bundle["features"],
                "Importance": model.feature_importances_,
            }).sort_values(
                "Importance", ascending=False
            ).head(15)

            st.bar_chart(
                importance.sort_values(
                    "Importance", ascending=True
                ),
                x="Importance",
                y="Feature",
            )

            st.caption(
                "Feature importance indicates how the model uses "
                "features; it does not establish causation."
            )

        else:
            st.info(
                "This model does not expose feature_importances_. "
                "Use SHAP or a compatible model explanation method "
                "to investigate individual feature contributions."
            )

    st.divider()
    st.subheader("Dataset Overview")

    overview1, overview2, overview3 = st.columns(3)

    overview1.metric(
        "Rows",
        f"{len(df):,}",
    )

    overview2.metric(
        "Columns",
        f"{len(df.columns):,}",
    )

    overview3.metric(
        "Missing Values",
        f"{int(df.isna().sum().sum()):,}",
    )

    st.subheader("Model Information")

    if model_bundle is not None:
        st.write(
            f"**Model type:** "
            f"{type(model_bundle['model']).__name__}"
        )
        st.write(
            f"**Expected features:** "
            f"{len(model_bundle['features'])}"
        )
        st.write(
            f"**Decision threshold:** "
            f"{model_bundle['threshold']:.2f}"
        )
    else:
        st.warning(model_error)

