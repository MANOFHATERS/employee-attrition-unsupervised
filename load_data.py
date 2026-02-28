# load_data.py - Section 1: Data Loading & Initial Exploration

import pandas as pd
import numpy as np

def load_and_explore(filepath='hr_employee_attrition.csv'):
    """Load the HR dataset and do initial exploration."""
    
    # loading the dataset
    df = pd.read_csv(filepath)
    
    # lets see what we're working with
    print("First 5 rows:")
    print(df.head())
    print(f"\nDataset Shape: {df.shape}")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    
    # checking the datatypes and non-null counts
    print("\nDataset Info:")
    print(df.info())
    
    # summary statistics
    print("\nDescriptive Statistics:")
    print(df.describe())
    
    # separate numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"\nNumeric columns ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    
    # checking for null values - important!
    print("\nNull values per column:")
    print(df.isnull().sum())
    print(f"\nTotal nulls: {df.isnull().sum().sum()}")  # should be 0
    
    # checking for duplicates
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate rows: {duplicates}")
    
    # value counts for categorical columns
    print("\n--- Value Counts for Categorical Columns ---")
    for col in categorical_cols:
        print(f"\n{col}:")
        print(df[col].value_counts())
    
    print("\n[OK] Section 1 complete - data loaded and explored")
    
    return df
