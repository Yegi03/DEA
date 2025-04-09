import pandas as pd
import openpyxl

def analyze_excel_structure(file_path):
    """Analyze the structure of an Excel file"""
    print(f"\nAnalyzing Excel file: {file_path}")
    
    # Read the Excel file
    df = pd.read_excel(file_path)
    
    print("\nFirst 20 rows of data:")
    print(df.head(20))
    
    print("\nShape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    
    print("\nSearching for potential header rows...")
    for idx, row in df.iterrows():
        # Convert row to string and check content
        row_str = ' '.join([str(val) for val in row.values])
        print(f"\nRow {idx}:", row_str[:100], "..." if len(row_str) > 100 else "")

def main():
    try:
        analyze_excel_structure('hospital_report_2023.xlsx')
    except Exception as e:
        print(f"Error analyzing file: {e}")

if __name__ == "__main__":
    main() 