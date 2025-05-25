import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os  # <--- import os here

def preprocess_data(df):
    # Fill missing values
    for col in ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Loan_Amount_Term', 'Credit_History']:
        df[col] = df[col].fillna(df[col].mode()[0])
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())

    # Encode categorical columns
    categorical_columns_for_encoding = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Amount_Term']
    label_encoders = {}
    for col in categorical_columns_for_encoding:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Scale numerical columns
    num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Credit_History']
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # Encode target variable
    target_encoder = LabelEncoder()
    df['Loan_Status'] = target_encoder.fit_transform(df['Loan_Status'])

    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)  # <--- add this here

    # Save target encoder for inference
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(target_encoder, f)

    # Prepare features and target
    X = df.drop(columns=['Loan_ID', 'Loan_Status'])
    y = df['Loan_Status']

    # Split dataset
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    return X_train, X_val, y_train, y_val, label_encoders, scaler
