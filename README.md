# Explainable Chat-Driven Loan Recommender System

This project implements an AI-powered system to predict loan approval decisions and provide users with clear, easy-to-understand explanations on why a loan was approved or rejected. It uses XGBoost for prediction, SMOTE for handling imbalanced data, and SHAP for generating feature importance explanations.

## Features

- Data preprocessing with handling missing values and encoding categorical variables
- Balances training data using SMOTE to improve model fairness
- Trains an XGBoost classifier for high-accuracy loan approval prediction
- Generates explanations for each decision using SHAP values
- Provides actionable advice for users based on model insights
- Visualizes feature contributions with pie charts and SHAP summary plots

## Setup

### Requirements

Make sure you have Python 3.7+ installed.

Install the required Python packages with:

```bash
pip install -r requirements.txt

## 4. Usage Instructions

This section explains how to run the loan approval recommender script, what output to expect, and how to interact with it.

### How to Run

1. **Ensure you have all required dependencies installed** (see `requirements.txt`).

2. **Run the main script** by opening a terminal or command prompt, navigating to the project directory, and typing:

```bash
python loan-recommender.py

## Credits

Built by Chandralekha using Python, SHAP, and XGBoost.

---

## How to Upload This Project to GitHub

### Step 4: Create a New GitHub Repository

1. Go to [https://github.com](https://github.com)
2. Click on **New Repository**
3. Give it a name like `loan-recommender`
4. (Optional) Initialize with README
5. Click **Create Repository**

---

### Step 5: Push Code to GitHub

Open terminal in your project folder and run:

```bash
git init
git remote add origin https://github.com/yChandralekha005/loan-recommender.git
git add .
git commit -m "Initial commit - Loan recommender system with SHAP explanations"
git branch -M main
git push -u origin main
