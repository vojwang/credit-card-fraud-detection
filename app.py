import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("credit_card_fraud_random_forest.pkl")

model = load_model()

# -----------------------------
# Title
# -----------------------------
st.title("💳 Credit Card Fraud Detection")

st.markdown("""
This application uses a **Random Forest Classifier** to detect fraudulent credit card transactions.

### Instructions
1. Upload a CSV file.
2. The file must contain the following columns:
- Time
- V1 to V28
- Amount

The model will classify every transaction as either:

- ✅ Legitimate
- 🚨 Fraudulent
""")

# -----------------------------
# Expected Columns
# -----------------------------
expected_columns = [
    "Time","V1","V2","V3","V4","V5","V6","V7","V8","V9",
    "V10","V11","V12","V13","V14","V15","V16","V17","V18",
    "V19","V20","V21","V22","V23","V24","V25","V26","V27",
    "V28","Amount"
]

# -----------------------------
# Upload CSV
# -----------------------------
uploaded_file = st.file_uploader(
    "📂 Upload Transaction CSV",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    # -----------------------------
    # Validate Columns
    # -----------------------------
    missing = [col for col in expected_columns if col not in data.columns]

    if len(missing) > 0:

        st.error("❌ Invalid dataset")

        st.write("Missing columns:")

        st.write(missing)

        st.stop()

    st.success("✅ Dataset uploaded successfully")

    st.subheader("Uploaded Data")

    st.dataframe(data.head())

    # -----------------------------
    # Prediction
    # -----------------------------
    predictions = model.predict(data)

    results = data.copy()

    results["Prediction"] = predictions

    results["Prediction"] = results["Prediction"].map({
        0: "Legitimate",
        1: "Fraudulent"
    })

    # -----------------------------
    # Summary
    # -----------------------------
    total = len(results)

    fraud = (results["Prediction"] == "Fraudulent").sum()

    legitimate = (results["Prediction"] == "Legitimate").sum()

    st.subheader("Prediction Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Transactions", total)

    col2.metric("Legitimate", legitimate)

    col3.metric("Fraudulent", fraud)

    # -----------------------------
    # Results
    # -----------------------------
    st.subheader("Prediction Results")

    st.dataframe(results)

    # -----------------------------
    # Download
    # -----------------------------
    csv = results.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Prediction Results",
        data=csv,
        file_name="fraud_predictions.csv",
        mime="text/csv"
    )
