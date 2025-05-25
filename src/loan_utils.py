import joblib
import pickle
import os

def save_label_encoders(label_encoders, folder='models'):
    os.makedirs(folder, exist_ok=True)
    for col, le in label_encoders.items():
        filename = f"{folder}/le_{col}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(le, f)

def load_label_encoders(cols, folder='models'):
    label_encoders = {}
    for col in cols:
        filename = f"{folder}/le_{col}.pkl"
        with open(filename, 'rb') as f:
            label_encoders[col] = pickle.load(f)
    return label_encoders

def save_scaler(scaler, folder='models'):
    os.makedirs(folder, exist_ok=True)
    joblib.dump(scaler, f"{folder}/scaler.pkl")

def load_scaler(folder='models'):
    return joblib.load(f"{folder}/scaler.pkl")

def save_model(model, folder='models'):
    os.makedirs(folder, exist_ok=True)
    joblib.dump(model, f"{folder}/xgb_model.pkl")

def load_model(folder='models'):
    return joblib.load(f"{folder}/xgb_model.pkl")

def save_explainer(explainer, folder='models'):
    os.makedirs(folder, exist_ok=True)
    joblib.dump(explainer, f"{folder}/shap_explainer.pkl")

def load_explainer(folder='models'):
    return joblib.load(f"{folder}/shap_explainer.pkl")

def save_feature_names(feature_names, folder='models'):
    os.makedirs(folder, exist_ok=True)
    with open(f"{folder}/feature_names.pkl", 'wb') as f:
        pickle.dump(feature_names, f)

def load_feature_names(folder='models'):
    with open(f"{folder}/feature_names.pkl", 'rb') as f:
        return pickle.load(f)
