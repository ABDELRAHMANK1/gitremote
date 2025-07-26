import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Function to load and prepare insurance data
def train_insurance_model(df):
    # Ensure no extra spaces in column names
    df.columns = df.columns.str.strip()

    # Handle categorical columns
    categorical_cols = ['sex', 'smoker', 'region']
    for col in categorical_cols:
        if col in df.columns:
            # Convert to string to avoid issues with LabelEncoder
            df[col] = df[col].astype(str)
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            # Save LabelEncoder for later prediction
            joblib.dump(le, f'saved_models/label_encoder_{col}.joblib')
        else:
            print(f"Warning: Column '{col}' not found in DataFrame for encoding.")

    # Define features (X) and target variable (y)
    features = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
    target = 'charges'

    # Ensure all required columns are present
    for col in features + [target]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in the dataset.")

    # Handle non-numeric values in numerical columns
    # Values that cannot be converted to numeric will become NaN
    for col in ['age', 'bmi', 'children', 'charges']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with NaN values in essential columns (features and target)
    df.dropna(subset=features + [target], inplace=True)

    X = df[features]
    y = df[target]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train RandomForestRegressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, mse, r2

# Prediction function
def predict_insurance_cost(model, data):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([data])

    # Apply same LabelEncoder transformations used during training
    categorical_cols = ['sex', 'smoker', 'region']
    for col in categorical_cols:
        if col in input_df.columns:
            le_path = f'saved_models/label_encoder_{col}.joblib'
            if os.path.exists(le_path):
                le = joblib.load(le_path)
                # Use transform only (without fit)
                input_df[col] = le.transform(input_df[col].astype(str))
            else:
                raise FileNotFoundError(f"LabelEncoder for '{col}' not found. Please train the model first.")
        else:
            raise ValueError(f"Input data missing expected column: '{col}'")

    # Ensure numerical columns are of the correct type
    for col in ['age', 'bmi', 'children']:
        if col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
            if input_df[col].isnull().any():
                raise ValueError(f"Numerical column '{col}' contains non-numeric values after conversion.")

    # Ensure column order matches the order used during model training
    features_order = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
    missing_features = [f for f in features_order if f not in input_df.columns]
    if missing_features:
        raise ValueError(f"Input data is missing the following features: {missing_features}")
    
    input_data_ordered = input_df[features_order]

    prediction = model.predict(input_data_ordered)
    return prediction[0]