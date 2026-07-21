import streamlit as st
import pandas as pd
import joblib

# ------------------------------------
# Page Configuration
# ------------------------------------

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# ------------------------------------
# Load Model
# ------------------------------------

model = joblib.load("credit_card_fraud_random_forest.pkl")

# ------------------------------------
# Title
# ------------------------------------

st.title("💳 Credit Card Fraud Detection")

st.write("""
This application uses a **Random Forest Classifier** to detect fraudulent credit card transactions.

Upload a CSV file containing transaction data, and the model will classify each transaction as either **Legitimate** or **Fraudulent**.
""")

# ------------------------------------
# Upload CSV
# ------------------------------------

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")

    st.dataframe(data.head())

    # Prediction
    predictions = model.predict(data)

    results = data.copy()
    results["Prediction"] = predictions

    results["Prediction"] = results["Prediction"].map({
        0: "Legitimate",
        1: "Fraudulent"
    })

    st.subheader("Prediction Results")

    st.dataframe(results)

    csv = results.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Results",
        csv,
        "fraud_predictions.csv",
        "text/csv"
    )