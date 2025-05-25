from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import joblib
import os

def train_model(X_train, y_train, X_val, y_val):
    model = XGBClassifier(
        eval_metric='logloss',
        random_state=42,
        reg_alpha=0.1,
        reg_lambda=1.0,
        gamma=0.1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    print("Validation Accuracy:", accuracy_score(y_val, y_pred))
    print(classification_report(y_val, y_pred))

    cm = confusion_matrix(y_val, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Validation Set")
    plt.savefig('models/confusion_matrix_val.png')
    plt.close()


    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    model.fit(X_train_sm, y_train_sm)

    y_pred_sm = model.predict(X_val)
    print("Validation Accuracy after SMOTE:", accuracy_score(y_val, y_pred_sm))
    print(classification_report(y_val, y_pred_sm))

    cm_sm = confusion_matrix(y_val, y_pred_sm)
    disp_sm = ConfusionMatrixDisplay(confusion_matrix=cm_sm, display_labels=model.classes_)
    disp_sm.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Validation Set after SMOTE")
    plt.savefig('models/confusion_matrix_val_smote.png')
    plt.close()


    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/xgb_model.pkl')
    print("Model saved successfully.")

    return model, X_train_sm, y_train_sm
