import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load saved model and scaler
model = joblib.load('rf_fraud_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("💳 Credit Card Fraud Detection")

st.markdown("Enter transaction details below to check if it's **Fraudulent** or **Legitimate**.")

feature_names = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9','V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17','V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25','V26', 'V27', 'V28', 'Amount']

user_input = []

for feature in feature_names:
    val = st.number_input(f"{feature}:", value=0.0)
    user_input.append(val)

if st.button("Predict"):
    input_np = np.array(user_input).reshape(1, -1)
    input_scaled = scaler.transform(input_np)
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Alert: This transaction is predicted to be **FRAUDULENT** with {prediction_proba:.2%} confidence.")
    else:
        st.success(f"✅ Safe: This transaction is predicted to be **LEGITIMATE** with {1 - prediction_proba:.2%} confidence.")
