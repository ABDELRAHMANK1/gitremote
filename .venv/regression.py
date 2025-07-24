import streamlit as st 
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeRegressor

@st.cache_data
def loud_data(file):
    return pd.read_csv(file)

file = st.file_uploader("Uploud the fiel",type=["csv"])


if file!=None:
    df=loud_data(file)
    model_type=st.selectbox("choose the algorithm to regression:",["Random forest regression","Decision tree regression","support vector regression"])
    if model_type=="Random forest regression":
        categorical_cols = ['sex', 'smoker', 'region']

        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            # Our target variable for regression is 'charges'
        X = df.drop('charges', axis=1)  # Features are all columns except 'charges'
        y = df['charges']      
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        # Create the Random Forest Regressor
        regressor = RandomForestRegressor(n_estimators=100, random_state=0)

        # Fit the model
        regressor.fit(X_train, y_train)

        y_pred = regressor.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write(f"Mean Squared Error (MSE): {mse:.2f}")
        st.write(f"R-squared (R²): {r2:.2f}")
    elif model_type=="support vector regression":
        label_encoders = {}
        for col in ['sex', 'smoker', 'region']:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

# Step 3: Define features (X) and target (y)
        X = df.drop('charges', axis=1)
        y = df['charges']

# Step 4: Scale features
        scaler_X = StandardScaler() 
        scaler_y = StandardScaler()

        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()  # Flatten to 1D

# Step 5: Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Step 6: Train SVR model
        svr = SVR(kernel='rbf')  
        svr.fit(X_train, y_train)

# Step 7: Predict and inverse scale
        y_pred_scaled = svr.predict(X_test)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
        y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1))

# Step 8: Evaluate
        mse = mean_squared_error(y_test_original, y_pred)
        r2 = r2_score(y_test_original, y_pred)

        st.write(f"Mean Squared Error: {mse:.2f}")
        st.write(f"R² Score: {r2:.2f}")
    elif model_type=="Decision tree regression":
        label_encoders = {}
        for col in ['sex', 'smoker', 'region']:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        x = df.drop('charges', axis=1)  # Features are all columns except 'charges'
        y = df['charges']               # Target is 'charges'
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        dr_regressor = DecisionTreeRegressor(random_state =46)
        dr_regressor.fit(X_train,y_train)
        y_pred = dr_regressor.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write("Decision Tree Regression Results:")
        st.write(f"Mean Squared Error (MSE): {mse:.2f}")
        st.write(f"R-squared (R²): {r2:.2f}")




