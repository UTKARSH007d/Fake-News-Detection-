import pandas as pd

# Load the dataset
df = pd.read_csv("data/raw/WELFake_Dataset.csv")

# Display basic information
print("=" * 60)
print("FAKE NEWS DATASET OVERVIEW")
print("=" * 60)

# Shape
print("\nDataset Shape:")
print(df.shape)

# Column names
print("\nColumns:")
print(df.columns.tolist())

# Data types
print("\nData Types:")
print(df.dtypes.to_string())

# First 5 rows
print("\nFirst 5 Rows:")
print(df.head())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum().to_string())

# Duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Label distribution
print("\nLabel Distribution:")
print(df["label"].value_counts().to_string())