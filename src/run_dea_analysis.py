import pandas as pd
import numpy as np
from data_preparation import DataPreparator
from dea_analysis import DEAAnalyzer
from visualization import DEAVisualizer
from advanced_analysis import AdvancedAnalyzer
import os
import json

def convert_to_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
        np.int16, np.int32, np.int64, np.uint8,
        np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

def main():
    # Initialize data preparation
    data_prep = DataPreparator()
    
    # Try to load prepared data from the root directory
    try:
        prepared_data_path = 'prepared_data.csv'  # Path relative to root directory
        print(f"Loading prepared data from: {prepared_data_path}")
        combined_data = pd.read_csv(prepared_data_path)
        print(f"Successfully loaded prepared data. Shape: {combined_data.shape}")
    except Exception as e:
        print(f"Error loading prepared data: {e}")
        return
    
    # Initialize DEA analyzer
    dea_analyzer = DEAAnalyzer()
    
    # Prepare DEA data
    print("\nPreparing DEA data...")
    dea_data = dea_analyzer.prepare_dea_data(combined_data)
    
    if dea_data is None:
        print("Error: Failed to prepare DEA data")
        return
    
    # Calculate technical efficiency (both VRS and CRS)
    dea_data = dea_analyzer.calculate_technical_efficiency(dea_data)
    if dea_data is None:
        print("Error: Failed to calculate technical efficiency")
        return
    
    # Calculate scale efficiency
    dea_data = dea_analyzer.calculate_scale_efficiency(dea_data)
    if dea_data is None:
        print("Error: Failed to calculate scale efficiency")
        return
    
    # Create hospital categories
    print("\nCreating hospital categories...")
    dea_data['Size_Category'] = pd.cut(dea_data['Licensed_Beds'], 
                                  bins=[0, 100, 300, float('inf')],
                                  labels=['Small', 'Medium', 'Large'],
                                  include_lowest=True)
    
    dea_data['Occupancy_Category'] = pd.cut(dea_data['Bed_Occupancy_Rate'],
                                       bins=[0, 40, 70, float('inf')],
                                       labels=['Low', 'Medium', 'High'],
                                       include_lowest=True)
    
    dea_data['Quality_Category'] = pd.cut(dea_data['Patient_Satisfaction'],
                                     bins=[0, 80, 90, float('inf')],
                                     labels=['Low', 'Medium', 'High'],
                                     include_lowest=True)
    
    # Perform context-dependent analysis
    dea_data = dea_analyzer.perform_context_dependent_analysis(dea_data)
    if dea_data is None:
        print("Error: Failed to perform context-dependent analysis")
        return
    
    # Save results
    output_file = 'dea_results.csv'
    dea_data.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Number of hospitals analyzed: {len(dea_data)}")
    print("\nTechnical Efficiency (VRS):")
    print(dea_data['TE_VRS'].describe())
    print("\nTechnical Efficiency (CRS):")
    print(dea_data['TE_CRS'].describe())
    print("\nScale Efficiency:")
    print(dea_data['Scale_Efficiency'].describe())
    print("\nScale Efficiency Categories:")
    print(dea_data['Scale_Efficiency_Category'].value_counts())
    
    # Validate results
    print("\nValidating results...")
    validation_results = dea_analyzer.validate_results(dea_data)
    
    # Save validation results
    with open('validation_results.txt', 'w') as f:
        validation_results = convert_to_serializable(validation_results)
        f.write(json.dumps(validation_results, indent=4))
    
    # Create basic visualizations
    print("\nGenerating basic visualizations...")
    visualizer = DEAVisualizer()
    visualizer.generate_all_visualizations(dea_data)
    
    # Perform advanced analysis
    print("\nPerforming advanced analysis...")
    advanced_analyzer = AdvancedAnalyzer()
    advanced_analyzer.run_advanced_analysis(dea_data)
    
    print("\nAnalysis completed successfully!")

if __name__ == "__main__":
    main() 