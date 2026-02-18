import pandas as pd

df = pd.read_excel("PCOS_data_without_infertility.xlsx")
print("Column names:")
print(df.columns.tolist())
print("\nFirst few rows:")
print(df.head())
