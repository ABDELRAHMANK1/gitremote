import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Function to load and prepare loan data
def train_loan_model(df):
    # Ensure no extra spaces in column names
    df.columns = df.columns.str.strip()

    # Handle categorical columns
    categorical_cols = ['person_gender', 'person_education', 'person_home_ownership', 'loan_intent', 'previous_loan_defaults_on_file']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str) # Convert to string to avoid issues with LabelEncoder
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            joblib.dump(le, f'saved_models/label_encoder_loan_{col}.joblib')
        else:
            print(f"Warning: Column '{col}' not found in DataFrame for encoding.")

    # Define features (X) and target variable (y)
    features = [
        'person_age', 'person_income', 'person_emp_exp', 'person_home_ownership',
        'person_education', 'person_gender', 'loan_intent', 'loan_amnt',
        'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
        'credit_score', 'previous_loan_defaults_on_file'
    ]
    target = 'loan_status' # 0 = No, 1 = Yes

    # Ensure all required columns are present
    for col in features + [target]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in the dataset.")

    # Handle non-numeric values in numerical columns
    numerical_cols = [
        'person_age', 'person_income', 'person_emp_exp', 'loan_amnt',
        'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score'
    ]
    for col in numerical_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with NaN values in essential columns
    df.dropna(subset=features + [target], inplace=True)
    
    # Ensure target variable is integer type
    df[target] = df[target].astype(int)

    X = df[features]
    y = df[target]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train RandomForestClassifier model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    return model, accuracy, report, cm

# Prediction function
def predict_loan_default(model, data):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([data])

    # Apply same LabelEncoder transformations used during training
    categorical_cols = ['person_gender', 'person_education', 'person_home_ownership', 'loan_intent', 'previous_loan_defaults_on_file']
    for col in categorical_cols:
        if col in input_df.columns:
            le_path = f'saved_models/label_encoder_loan_{col}.joblib'
            if os.path.exists(le_path):
                le = joblib.load(le_path)
                input_df[col] = le.transform(input_df[col].astype(str))
            else:
                raise FileNotFoundError(f"LabelEncoder for loan '{col}' not found. Please train the model first.")
        else:
            raise ValueError(f"Input data missing expected column: '{col}'")

    # Ensure numerical columns are of the correct type
    numerical_cols = [
        'person_age', 'person_income', 'person_emp_exp', 'loan_amnt',
        'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score'
    ]
    for col in numerical_cols:
        if col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
            if input_df[col].isnull().any():
                raise ValueError(f"Numerical column '{col}' contains non-numeric values after conversion.")

    # Ensure column order matches the order used during model training
    features_order = [
        'person_age', 'person_income', 'person_emp_exp', 'person_home_ownership',
        'person_education', 'person_gender', 'loan_intent', 'loan_amnt',
        'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
        'credit_score', 'previous_loan_defaults_on_file'
    ]
    missing_features = [f for f in features_order if f not in input_df.columns]
    if missing_features:
        raise ValueError(f"Input data is missing the following features: {missing_features}")

    input_data_ordered = input_df[features_order]

    prediction = model.predict(input_data_ordered)
    return prediction[0]