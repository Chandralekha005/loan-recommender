import shap
import pandas as pd
import matplotlib.pyplot as plt

feature_explanations = {
    'Credit_History': "Your credit history is a major factor. A poor or missing credit history negatively affects approval.",
    'LoanAmount': "The loan amount you requested might be too high relative to your income.",
    'ApplicantIncome': "Your income level affects your ability to repay the loan.",
    'CoapplicantIncome': "If you have a co-applicant with sufficient income, it helps your application.",
    'Loan_Amount_Term': "The loan repayment term impacts your monthly payments and approval chances.",
    'Property_Area': "The location of the property can influence loan decisions.",
    'Dependents': "Number of dependents affects your financial obligations.",
    'Education': "Your education level may influence lender confidence.",
    'Self_Employed': "Being self-employed can affect your income stability assessment.",
    'Gender': "Gender may have indirect influences in credit decisions.",
    'Married': "Marital status might be considered for loan approval.",
}

def shap_explanation_for_sample(shap_values, X_val, sample_index):
    feature_names = X_val.columns
    sample_shap_values = shap_values[sample_index]
    sample_features = X_val.iloc[sample_index]

    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'SHAP Value': sample_shap_values,
        'Feature Value': sample_features
    }).sort_values(by='SHAP Value', key=abs, ascending=False)

    return shap_df

def generate_layman_text(shap_df):
    top_features = shap_df.head(3)
    messages = []
    for _, row in top_features.iterrows():
        feat = row['Feature']
        val = row['Feature Value']
        impact = row['SHAP Value']

        if feat == 'Credit_History':
            if val == 0:
                explanation = "Your credit history is missing or poor, which strongly decreases your chance."
            else:
                explanation = "You have a good credit history, positively influencing your application."
        elif feat == 'LoanAmount':
            if val > 0.5:
                explanation = "The loan amount you requested is quite high compared to typical amounts, which may reduce approval chances."
            else:
                explanation = "Your requested loan amount is reasonable."
        elif feat == 'ApplicantIncome':
            if val < 0:
                explanation = "Your income is lower than average, which might limit repayment ability."
            else:
                explanation = "Your income is healthy, helping your loan approval."
        else:
            explanation = feature_explanations.get(feat, f"{feat} affects your loan decision.")

        direction = "negatively" if impact < 0 else "positively"
        messages.append(f"{explanation} This feature influenced your loan application {direction} with a value of {val:.2f}.")

    return "\n".join(messages)

def generate_advice(shap_df):
    negative_feats = shap_df[shap_df['SHAP Value'] < 0]
    advices = []
    for _, row in negative_feats.iterrows():
        feat = row['Feature']
        if feat == 'Credit_History':
            advices.append("Try to improve your credit score or provide proof of past good credit.")
        elif feat == 'LoanAmount':
            advices.append("Consider requesting a smaller loan amount to increase chances of approval.")
        elif feat == 'ApplicantIncome':
            advices.append("Increasing your income or showing additional sources of income can help.")
        elif feat == 'CoapplicantIncome':
            advices.append("Adding a co-applicant with a stable income may improve your chances.")
        elif feat == 'Loan_Amount_Term':
            advices.append("Consider selecting a longer repayment term to reduce EMI burden.")
        elif feat == 'Self_Employed':
            advices.append("Provide financial records to show income stability if self-employed.")
        elif feat == 'Dependents':
            advices.append("If possible, reduce financial obligations or highlight other support sources.")
        elif feat == 'Education':
            advices.append("Showcase additional qualifications or financial literacy if applicable.")
        elif feat == 'Property_Area':
            advices.append("Highlight value or resale potential of the property to strengthen your case.")
        elif feat == 'Gender':
            advices.append("Ensure all other factors are strong to mitigate bias, if any.")
        elif feat == 'Married':
            advices.append("Provide documentation about dual income if applicable.")
        else:
            advices.append(f"Consider improving or clarifying your {feat.lower().replace('_', ' ')}.")

    if not advices:
        advices.append("You're doing great! Try contacting the lender to clarify any specific concerns.")

    return advices

def plot_feature_contributions(shap_df):
    plt.figure(figsize=(7, 7))
    plt.pie(shap_df['SHAP Value'].abs(), labels=shap_df['Feature'], autopct='%1.1f%%')
    plt.title("Feature Contribution to Loan Decision")
    plt.show()

def create_loan_report(model, X_val, shap_values, sample_index):
    shap_df = shap_explanation_for_sample(shap_values, X_val, sample_index)
    layman_text = generate_layman_text(shap_df)
    advice = generate_advice(shap_df)

    pred = model.predict(X_val.iloc[[sample_index]])[0]
    status = "Approved" if pred == 1 else "Denied"

    print(f"Loan Application Status: {status}")
    print("\nDetailed Explanation:")
    print(layman_text)
    print("\nPersonalized Advice:")
    for a in advice:
        print(f"- {a}")

    plot_feature_contributions(shap_df)
