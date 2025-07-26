import streamlit as st
import pandas as pd
import joblib
import os
from regression_model import train_insurance_model, predict_insurance_cost
from classification_model import train_loan_model, predict_loan_default
import numpy as np # For handling potential NaN values if needed

# Define model save directory
MODEL_DIR = 'saved_models'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

st.set_page_config(layout="wide", page_title="Medical & Loan Prediction App")

st.title("Medical & Loan Prediction Application 📊")

# Project type selection (sidebar)
project_type = st.sidebar.radio(
    "Select Prediction Type:",
    ("Predict Health Insurance Cost", "Predict Loan Default Risk")
)

# -----------------------------------------------------------------------------
#                   Health Insurance Cost Prediction (Regression)
# -----------------------------------------------------------------------------
if project_type == "Predict Health Insurance Cost":
    st.header("🩺 Predict Health Insurance Cost")

    st.subheader("1. Train Insurance Model")
    uploaded_insurance_file = st.file_uploader("Upload your 'medical_insurance.csv' file for training:", type=["csv"], key="insurance_upload")

    if uploaded_insurance_file is not None:
        if st.button("Train and Save Insurance Model", key="train_insurance_button"):
            try:
                # Read the uploaded file into a DataFrame
                df_insurance = pd.read_csv(uploaded_insurance_file)
                st.write("Uploaded Insurance Data Preview:")
                st.dataframe(df_insurance.head())

                model, mse, r2 = train_insurance_model(df_insurance)
                
                joblib.dump(model, os.path.join(MODEL_DIR, 'insurance_model.joblib'))
                st.success("Insurance model trained and saved successfully!")
                st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
                st.write(f"**R-squared (R²):** {r2:.2f}")
            except Exception as e:
                st.error(f"Error training insurance model: {e}")
    else:
        st.info("Please upload a 'medical_insurance.csv' file to train the model.")

    # Prediction section
    st.subheader("2. Predict Cost for New Data:")
    insurance_model_path = os.path.join(MODEL_DIR, 'insurance_model.joblib')

    if os.path.exists(insurance_model_path):
        insurance_model = joblib.load(insurance_model_path)

        with st.form("insurance_prediction_form"):
            st.write("Enter patient details:")
            age = st.slider("Age:", min_value=18, max_value=100, value=30)
            sex = st.selectbox("Sex:", ["male", "female"])
            bmi = st.slider("Body Mass Index (BMI):", min_value=10.0, max_value=60.0, value=25.0, format="%.2f")
            children = st.slider("Number of Children:", min_value=0, max_value=10, value=0)
            smoker = st.selectbox("Smoker?", ["yes", "no"])
            region = st.selectbox("Region:", ["southwest", "southeast", "northwest", "northeast"])

            submitted_insurance = st.form_submit_button("Predict Insurance Cost")

            if submitted_insurance:
                try:
                    input_data = {
                        'age': age,
                        'sex': sex,
                        'bmi': bmi,
                        'children': children,
                        'smoker': smoker,
                        'region': region
                    }
                    prediction = predict_insurance_cost(insurance_model, input_data)
                    st.success(f"Predicted Health Insurance Cost: ${prediction:,.2f}")
                except Exception as e:
                    st.error(f"Error during insurance prediction: {e}")
    else:
        st.warning("Insurance model not trained yet. Please upload a file and click 'Train and Save Insurance Model' first.")

# -----------------------------------------------------------------------------
#                   Loan Default Risk Prediction (Classification)
# -----------------------------------------------------------------------------
elif project_type == "Predict Loan Default Risk":
    st.header("🏦 Predict Loan Default Risk")

    st.subheader("1. Train Loan Model")
    uploaded_loan_file = st.file_uploader("Upload your 'loan_data.csv' file for training:", type=["csv"], key="loan_upload")

    if uploaded_loan_file is not None:
        if st.button("Train and Save Loan Model", key="train_loan_button"):
            try:
                # Read the uploaded file into a DataFrame
                df_loan = pd.read_csv(uploaded_loan_file)
                st.write("Uploaded Loan Data Preview:")
                st.dataframe(df_loan.head())

                model, accuracy, report, cm = train_loan_model(df_loan)
                
                joblib.dump(model, os.path.join(MODEL_DIR, 'loan_model.joblib'))
                st.success("Loan model trained and saved successfully!")
                st.write(f"**Model Accuracy:** {accuracy:.2f}")
                st.subheader("Classification Report:")
                st.json(report) # Display report as JSON for readability
                st.subheader("Confusion Matrix:")
                st.code(cm.tolist()) # Display confusion matrix as a list

            except Exception as e:
                st.error(f"Error training loan model: {e}")
    else:
        st.info("Please upload a 'loan_data.csv' file to train the model.")

    # Prediction section
    st.subheader("2. Predict Loan Status for New Data:")
    loan_model_path = os.path.join(MODEL_DIR, 'loan_model.joblib')

    if os.path.exists(loan_model_path):
        loan_model = joblib.load(loan_model_path)

        with st.form("loan_prediction_form"):
            st.write("Enter loan applicant details:")
            col1, col2 = st.columns(2)
            with col1:
                person_age = st.slider("Borrower Age:", min_value=18, max_value=100, value=30)
                person_income = st.slider("Annual Income:", min_value=0.0, value=50000.0, format="%.2f")
                person_emp_exp = st.slider("Employment Experience (years):", min_value=0, max_value=60, value=5)
                person_home_ownership = st.selectbox("Home Ownership:", ["RENT", "OWN", "MORTGAGE", "OTHER"])
                person_education = st.selectbox("Education Level:", ["Bachelor", "Master", "High School", "Associate", "PhD"])
                person_gender = st.selectbox("Gender:", ["male", "female"])
                loan_intent = st.selectbox("Loan Intent:", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
            with col2:
                loan_amnt = st.slider("Loan Amount:", min_value=1.0, value=10000.0, format="%.2f")
                loan_int_rate = st.slider("Interest Rate (%):", min_value=0.0, max_value=30.0, value=10.0, format="%.2f")
                loan_percent_income = st.slider("Loan to Income Ratio (%):", min_value=0.0, max_value=1.0, value=0.1, format="%.2f")
                cb_person_cred_hist_length = st.slider("Credit History Length (years):", min_value=0, max_value=50, value=2)
                credit_score = st.slider("Credit Score:", min_value=300, max_value=850, value=650)
                previous_loan_defaults_on_file = st.selectbox("Previous Loan Defaults on File?:", ["No", "Yes"])

            submitted_loan = st.form_submit_button("Predict Loan Status")

            if submitted_loan:
                try:
                    input_data = {
                        'person_age': person_age,
                        'person_income': person_income,
                        'person_emp_exp': person_emp_exp,
                        'person_home_ownership': person_home_ownership,
                        'person_education': person_education,
                        'person_gender': person_gender,
                        'loan_intent': loan_intent,
                        'loan_amnt': loan_amnt,
                        'loan_int_rate': loan_int_rate,
                        'loan_percent_income': loan_percent_income,
                        'cb_person_cred_hist_length': cb_person_cred_hist_length,
                        'credit_score': credit_score,
                        'previous_loan_defaults_on_file': previous_loan_defaults_on_file
                    }
                    prediction = predict_loan_default(loan_model, input_data)
                    if prediction == 1:
                        st.error("⚠️ **Prediction: Customer WILL default on loan.**")
                    else:
                        st.success("✅ **Prediction: Customer WILL NOT default on loan.**")
                except Exception as e:
                    st.error(f"Error during loan prediction: {e}")
    else:
        st.warning("Loan model not trained yet. Please upload a file and click 'Train and Save Loan Model' first.")