import pandas as pd

def load_data():
    file_path = r"C:\Users\DELL\Documents\loan-recommender\data\train_ctrUa4K.csv"
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully!")
    print("Shape of the dataset (rows, columns):", df.shape)
    print("\nMissing values in each column:")
    print(df.isnull().sum())
    return df
