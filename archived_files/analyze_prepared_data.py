import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_dataframe(df, name):
    """Analyze a DataFrame and print detailed information"""
    print(f"\n{'='*50}")
    print(f"Analysis of {name}")
    print(f"{'='*50}")
    
    # Basic information
    print(f"\nShape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Data types
    print("\nData Types:")
    print(df.dtypes)
    
    # Missing values
    print("\nMissing Values:")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.any() else "No missing values")
    
    # Descriptive statistics
    print("\nDescriptive Statistics:")
    print(df.describe())
    
    # Correlation matrix
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        print("\nCorrelation Matrix:")
        corr = df[numeric_cols].corr()
        print(corr)
        
        # Create correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
        plt.title(f'Correlation Matrix - {name}')
        plt.tight_layout()
        plt.savefig(f'correlation_{name.lower().replace(" ", "_")}.png')
        plt.close()
    
    # Value distributions
    for col in numeric_cols:
        plt.figure(figsize=(8, 4))
        sns.histplot(data=df, x=col)
        plt.title(f'Distribution of {col}')
        plt.tight_layout()
        plt.savefig(f'distribution_{col.lower().replace(" ", "_")}.png')
        plt.close()

def main():
    # Load prepared data
    try:
        prepared_df = pd.read_csv('prepared_data.csv')
        inputs_df = pd.read_csv('dea_inputs.csv')
        outputs_df = pd.read_csv('dea_outputs.csv')
        
        # Analyze each dataset
        analyze_dataframe(prepared_df, "Prepared Data")
        analyze_dataframe(inputs_df, "Input Variables")
        analyze_dataframe(outputs_df, "Output Variables")
        
        # Additional analysis
        print("\nAdditional Analysis:")
        
        # Hospital count by input/output combinations
        print("\nHospitals by Input/Output Combinations:")
        input_counts = inputs_df.apply(lambda x: tuple(x.round(2)), axis=1).value_counts()
        output_counts = outputs_df.apply(lambda x: tuple(x.round(2)), axis=1).value_counts()
        
        print("\nUnique Input Combinations:", len(input_counts))
        print("Most common input combinations:")
        print(input_counts.head())
        
        print("\nUnique Output Combinations:", len(output_counts))
        print("Most common output combinations:")
        print(output_counts.head())
        
        # Efficiency potential analysis
        print("\nEfficiency Potential Analysis:")
        for col in outputs_df.columns:
            print(f"\n{col}:")
            print(f"Min: {outputs_df[col].min():.2f}")
            print(f"Max: {outputs_df[col].max():.2f}")
            print(f"Mean: {outputs_df[col].mean():.2f}")
            print(f"Std: {outputs_df[col].std():.2f}")
        
    except Exception as e:
        print(f"Error analyzing data: {e}")

if __name__ == "__main__":
    main() 