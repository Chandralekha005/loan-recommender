from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.train_model import train_model
from src.shap_explainer import create_loan_report
import shap

def main():
    # Load and preprocess data
    df = load_data()
    X_train, X_val, y_train, y_val, label_encoders, scaler = preprocess_data(df)
    
    # Save encoders and scaler for future inference
        # Save trained model, explainer, and feature names
    
    # Train model with SMOTE handling
    model, X_train_sm, y_train_sm = train_model(X_train, y_train, X_val, y_val)

    # SHAP explainability
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)

    # Generate report for a sample
    sample_index = 0  # Change index as needed
    create_loan_report(model, X_val, shap_values, sample_index)
    
    from src.loan_utils import save_model, save_explainer, save_feature_names

    save_model(model)
    save_explainer(explainer)
    save_feature_names(X_val.columns.tolist())

if __name__ == "__main__":
    main()
