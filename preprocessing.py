# preprocessing.py - Sections 3, 4, 5: Data Cleaning, Encoding, Scaling

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def clean_and_preprocess(df):
    """Clean the data, encode categoricals, and scale features."""
    
    # ============================================================
    # SECTION 3 — Data Cleaning
    # ============================================================
    
    print(f"\nShape before cleaning: {df.shape}")
    
    # these columns have only 1 unique value - totally useless
    # EmployeeCount is always 1, StandardHours always 80, Over18 always Y
    print(f"EmployeeCount unique: {df['EmployeeCount'].nunique()} -> {df['EmployeeCount'].unique()}")
    print(f"StandardHours unique: {df['StandardHours'].nunique()} -> {df['StandardHours'].unique()}")
    print(f"Over18 unique: {df['Over18'].nunique()} -> {df['Over18'].unique()}")
    
    # dropping them + EmployeeNumber (just an ID)
    cols_to_drop = ['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber']
    df_clean = df.drop(columns=cols_to_drop)
    print(f"\nDropped columns: {cols_to_drop}")
    print(f"Shape after cleaning: {df_clean.shape}")
    
    # confirming no nulls after cleaning
    print(f"Null values after cleaning: {df_clean.isnull().sum().sum()}")
    
    print("\n[OK] Section 3 complete - data cleaning done")
    
    # ============================================================
    # SECTION 4 — Data Preprocessing & Encoding
    # ============================================================
    
    # saving attrition for later analysis - need this!
    attrition_labels = df_clean['Attrition'].copy()
    
    print(f"\nShape before encoding: {df_clean.shape}")
    
    # binary encoding for columns with 2 values
    df_encoded = df_clean.copy()
    df_encoded['Gender'] = np.where(df_encoded['Gender'] == 'Female', 1, 0)
    df_encoded['Attrition'] = np.where(df_encoded['Attrition'] == 'Yes', 1, 0)
    df_encoded['OverTime'] = np.where(df_encoded['OverTime'] == 'Yes', 1, 0)
    print("Binary encoded: Gender (Female=1), Attrition (Yes=1), OverTime (Yes=1)")
    
    # one-hot encoding for multi-category columns
    multi_cat_cols = ['Department', 'BusinessTravel', 'EducationField',
                      'JobRole', 'MaritalStatus']
    df_encoded = pd.get_dummies(df_encoded, columns=multi_cat_cols, drop_first=False)
    print(f"One-hot encoded: {multi_cat_cols}")
    
    # saving attrition column and removing from modeling df
    attrition_col = df_encoded['Attrition'].copy()
    df_model = df_encoded.drop(columns=['Attrition'])
    
    print(f"\nShape after encoding: {df_model.shape}")
    print(f"Columns ({len(df_model.columns)}): {list(df_model.columns)}")
    # went from 31 columns to a lot more because of one-hot encoding
    # thats expected - each category becomes its own binary column
    
    print("\n[OK] Section 4 complete - preprocessing done")
    
    # ============================================================
    # SECTION 5 — Feature Scaling
    # ============================================================
    
    # scaling is super important for KMeans - it uses distances
    # without scaling, features with large ranges dominate
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_model),
                             columns=df_model.columns)
    
    # verifying the scaling worked
    print("\nScaling verification:")
    print(f"Mean of all columns (should be ~0): {df_scaled.mean().mean():.6f}")
    print(f"Std of all columns (should be ~1): {df_scaled.std().mean():.4f}")
    
    print("\n[OK] Section 5 complete - feature scaling done")
    
    return df_clean, df_encoded, df_model, df_scaled, attrition_labels, attrition_col
