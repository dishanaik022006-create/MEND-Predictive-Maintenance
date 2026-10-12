
import pandas as pd
from pathlib import Path

# Find the dataset in the project folder
file_path = Path("ai4i2020.csv")

if not file_path.exists():
    print("ERROR: Dataset file not found!")
    print("Make sure ai4i2020.csv is inside your project folder.")
else:
    data = pd.read_csv(file_path)

    print("\nDATASET LOADED SUCCESSFULLY!")
    print("-" * 40)

    print("\nFirst 5 rows:")
    print(data.head())

    print("\nDataset shape (rows, columns):")
    print(data.shape)

    print("\nColumn names:")
    print(data.columns.tolist())

    print("\nMissing values in each column:")
    print(data.isnull().sum())

    print("\nDataset information:")
    data.info()
