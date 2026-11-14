"""
Customer Churn Risk Predictor – Streamlit app
Loads a pre-trained Logistic Regression pipeline (StandardScaler + LogisticRegression)
and scores a single customer. No training happens in this app.
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "logistic_regression.joblib"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_FEATURES = [
    "Contract",
    "InternetService",
    "PaymentMethod",
    "PaperlessBilling",
    "TechSupport",
    "OnlineSecurity",
]

# Same risk bands used in the notebook
HIGH_THRESHOLD = 0.5
MEDIUM_THRESHOLD = 0.3

# Fallback column order (alphabetical categories, as produced by pd.get_dummies
# in the notebook). Only used if the model does not store feature_names_in_.
FALLBACK_COLUMNS = [
    "tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen",
    "Contract_Month-to-month", "Contract_One year", "Contract_Two year",
    "InternetService_DSL", "InternetService_Fiber optic", "InternetService_No",
    "PaymentMethod_Bank transfer (automatic)", "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check", "PaymentMethod_Mailed check",
    "PaperlessBilling_No", "PaperlessBilling_Yes",
    "TechSupport_No", "TechSupport_No internet service", "TechSupport_Yes",
    "OnlineSecurity_No", "OnlineSecurity_No internet service", "OnlineSecurity_Yes",
]


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return joblib.load(MODEL_PATH)


def get_expected_columns(model):
    cols = getattr(model, "feature_names_in_", None)
    if cols is None and hasattr(model, "steps"):  # sklearn Pipeline
        cols = getattr(model.steps[-1][1], "feature_names_in_", None)
        if cols is None:
            cols = getattr(model.steps[0][1], "feature_names_in_", None)
    return list(cols) if cols is not None else FALLBACK_COLUMNS


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
def build_feature_row(raw: dict, expected_columns: list) -> pd.DataFrame:
    """Replicate the notebook's preprocessing: one-hot encode and align columns."""
    df = pd.DataFrame([raw])
    df = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, dtype=int)
    return df.reindex(columns=expected_columns, fill_value=0)


def risk_level(probability: float) -> str:
    if probability >= HIGH_THRESHOLD:
        return "HIGH"
    if probability >= MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.set_page_config(page_title="Customer Churn Risk Predictor", page_icon="📉", layout="centered")

st.title("📉 Customer Churn Risk Predictor")
st.caption(
    "Estimate how likely a telecom customer is to leave. "
    "Powered by a pre-trained Logistic Regression model (Telco Customer Churn dataset)."
)

if not MODEL_PATH.exists():
    st.error(
        f"Model file not found: `{MODEL_PATH.name}`. "
        "Make sure `logistic_regression.joblib` is in the same folder as `app.py`."
    )
    st.stop()

try:
    model = load_model()
except Exception as exc:  # version mismatch, corrupt file, etc.
    st.error(
        "The model could not be loaded. This usually means the scikit-learn version "
        "in `requirements.txt` differs from the one used in Colab."
    )
    st.exception(exc)
    st.stop()

expected_columns = get_expected_columns(model)

st.subheader("Customer details")

col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12, step=1)
    monthly_charges = st.number_input(
        "Monthly charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=1.0
    )
    contract = st.selectbox("Contract type", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox(
        "Payment method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )

with col2:
    internet_service = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    senior = st.selectbox("Senior citizen", ["No", "Yes"])

    if internet_service == "No":
        tech_support = "No internet service"
        online_security = "No internet service"
        st.selectbox("Tech support", ["No internet service"], disabled=True)
        st.selectbox("Online security", ["No internet service"], disabled=True)
    else:
        tech_support = st.selectbox("Tech support", ["No", "Yes"])
        online_security = st.selectbox("Online security", ["No", "Yes"])

estimated_total = float(tenure * monthly_charges)
override = st.checkbox("Enter total charges manually", value=False)
if override:
    total_charges = st.number_input(
        "Total charges ($)", min_value=0.0, value=estimated_total, step=10.0
    )
else:
    total_charges = estimated_total
    st.caption(f"Total charges estimated as tenure × monthly charges = ${total_charges:,.2f}")

if st.button("Predict churn risk", type="primary", use_container_width=True):
    raw = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Contract": contract,
        "InternetService": internet_service,
        "PaymentMethod": payment_method,
        "PaperlessBilling": paperless,
        "TechSupport": tech_support,
        "OnlineSecurity": online_security,
    }

    features_df = build_feature_row(raw, expected_columns)
    probability = float(model.predict_proba(features_df)[0][1])
    level = risk_level(probability)

    st.divider()
    st.subheader("Result")

    m1, m2 = st.columns(2)
    m1.metric("Churn probability", f"{probability * 100:.1f}%")
    m2.metric("Risk level", level)
    st.progress(min(max(probability, 0.0), 1.0))

    if level == "HIGH":
        st.error("High risk: this customer is likely to leave. Prioritise for retention outreach.")
    elif level == "MEDIUM":
        st.warning("Medium risk: worth monitoring and considering a retention offer.")
    else:
        st.success("Low risk: this customer is unlikely to leave right now.")

    with st.expander("Model input (after encoding)"):
        st.dataframe(features_df.T.rename(columns={0: "value"}))

st.divider()
st.caption(
    f"Risk bands: LOW < {MEDIUM_THRESHOLD:.0%} ≤ MEDIUM < {HIGH_THRESHOLD:.0%} ≤ HIGH. "
    "Predictions are probabilities from a statistical model, not guarantees."
)
