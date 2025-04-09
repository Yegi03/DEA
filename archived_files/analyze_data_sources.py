import pandas as pd
import numpy as np
import openpyxl
import warnings
import os
warnings.filterwarnings('ignore')

def analyze_dataframe(df, source_name):
    """Analyze a DataFrame and print its structure"""
    print(f"\n{'='*50}")
    print(f"Analysis of {source_name}")
    print(f"{'='*50}")
    
    if df is None:
        print("Failed to load data")
        return
    
    print("\nBasic Information:")
    print(f"Shape: {df.shape}")
    print(f"Number of columns: {len(df.columns)}")
    print(f"Number of rows: {df.shape[0]}")
    
    print("\nColumns and Data Types:")
    for col in df.columns:
        print(f"- {col}: {df[col].dtype}")
        # Print sample values
        unique_values = df[col].nunique()
        print(f"  Unique values: {unique_values}")
        if unique_values < 10:
            print(f"  Sample values: {df[col].unique()[:5]}")
        else:
            print(f"  Sample values: {df[col].head(3).values}")
    
    print("\nMissing Values:")
    missing = df.isnull().sum()
    if missing.any():
        print(missing[missing > 0])
    else:
        print("No missing values")
    
    print("\nSummary Statistics:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if not numeric_cols.empty:
        print(df[numeric_cols].describe())

def main():
    # Analyze HCAHPS data
    if os.path.exists('HCAHPS-Patient-Care-Survey.csv'):
        try:
            hcahps_df = pd.read_csv('HCAHPS-Patient-Care-Survey.csv')
            analyze_dataframe(hcahps_df, "HCAHPS Survey Data")
        except Exception as e:
            print(f"Error loading HCAHPS data: {e}")
    else:
        print("HCAHPS data file not found.")
    
    # Analyze hospital report
    if os.path.exists('hospital_report_2023.xlsx'):
        try:
            hospital_df = pd.read_excel('hospital_report_2023.xlsx')
            analyze_dataframe(hospital_df, "Hospital Report Data")
        except Exception as e:
            print(f"Error loading hospital report: {e}")
    else:
        print("Hospital report file not found.")
    
    # Analyze compiled data
    if os.path.exists('compiled_data.csv'):
        try:
            compiled_df = pd.read_csv('compiled_data.csv')
            analyze_dataframe(compiled_df, "Compiled Data")
        except Exception as e:
            print(f"Error loading compiled data: {e}")
    else:
        print("Compiled data file not found.")
    
    # Analyze prepared data
    if os.path.exists('prepared_data.csv'):
        try:
            prepared_df = pd.read_csv('prepared_data.csv')
            analyze_dataframe(prepared_df, "Prepared Data")
        except Exception as e:
            print(f"Error loading prepared data: {e}")
    else:
        print("Prepared data file not found.")

if __name__ == "__main__":
    main() 