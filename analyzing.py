import pandas as pd


file_path = 'NVMe_Drive_Failure_Dataset.csv'
df = pd.read_csv(file_path)


print("--- First 5 Rows of the Dataset ---")
print(df.head())
print("\n" + "="*50 + "\n")


print("--- Dataset Information (Columns, Null Counts, Data Types) ---")
df.info()
print("\n" + "="*50 + "\n")


print("--- Summary Statistics ---")
print(df.describe())