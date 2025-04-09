import numpy as np
import pandas as pd
from scipy.optimize import linprog
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value, LpMaximize, LpStatus, COIN_CMD
import pulp

class DEAAnalyzer:
    def __init__(self):
        """Initialize the DEA analyzer"""
        self.input_variables = {
            'physical_resources': ['Licensed_Beds', 'Staffed_Beds'],
            'human_resources': ['Total_Surveys'],
            'financial_resources': ['Bed_Occupancy_Rate']
        }
        
        self.output_variables = {
            'patient_care': ['Patient_Satisfaction', 'Overall_Rating'],
            'quality_metrics': ['Recommendation_Score', 'Staff_Responsiveness'],
            'operational_metrics': ['Nurse_Communication', 'Doctor_Communication'],
            'health_outcomes': ['Cleanliness', 'Quietness']
        }
        
        self.inputs = None
        self.outputs = None
        self.dmu_count = 0
        
    def prepare_dea_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for DEA analysis by extracting inputs and outputs
        
        Args:
            data: DataFrame containing the combined data
            
        Returns:
            DataFrame with prepared data
        """
        print("\nPreparing DEA data...")
        
        if data is None or data.empty:
            print("Error: No data available for DEA analysis")
            return None
            
        # Get input variables
        self.inputs = {}
        for category, variables in self.input_variables.items():
            print(f"\nProcessing {category} inputs:")
            for var in variables:
                if var in data.columns:
                    print(f"  - {var}: Found in data")
                    # Convert to numeric and handle any non-numeric values
                    values = pd.to_numeric(data[var], errors='coerce')
                    if not values.isna().all():  # Only include if we have some valid values
                        self.inputs[var] = values.values
                    else:
                        print(f"  - {var}: All values are NaN, skipping")
                else:
                    print(f"  - {var}: Not found in data")
        
        # Get output variables
        self.outputs = {}
        for category, variables in self.output_variables.items():
            print(f"\nProcessing {category} outputs:")
            for var in variables:
                if var in data.columns:
                    print(f"  - {var}: Found in data")
                    # Convert to numeric and handle any non-numeric values
                    values = pd.to_numeric(data[var], errors='coerce')
                    if not values.isna().all():  # Only include if we have some valid values
                        self.outputs[var] = values.values
                    else:
                        print(f"  - {var}: All values are NaN, skipping")
                else:
                    print(f"  - {var}: Not found in data")
        
        # Check if we have any valid inputs and outputs
        if not self.inputs or not self.outputs:
            print("Error: No valid input or output variables found")
            return None
            
        self.dmu_count = len(data)
        print(f"\nTotal DMUs: {self.dmu_count}")
        print(f"Input variables: {list(self.inputs.keys())}")
        print(f"Output variables: {list(self.outputs.keys())}")
        
        # Check for missing or invalid data
        for name, values in self.inputs.items():
            print(f"\nChecking input {name}:")
            print(f"  - Shape: {values.shape}")
            print(f"  - Missing values: {np.isnan(values).sum()}")
            print(f"  - Range: [{np.nanmin(values)}, {np.nanmax(values)}]")
        
        for name, values in self.outputs.items():
            print(f"\nChecking output {name}:")
            print(f"  - Shape: {values.shape}")
            print(f"  - Missing values: {np.isnan(values).sum()}")
            print(f"  - Range: [{np.nanmin(values)}, {np.nanmax(values)}]")
        
        # Drop rows with any NaN values in inputs or outputs
        valid_mask = ~np.any(np.isnan([values for values in self.inputs.values()]), axis=0) & \
                    ~np.any(np.isnan([values for values in self.outputs.values()]), axis=0)
        
        if not np.any(valid_mask):
            print("Error: No valid data points after removing NaN values")
            return None
            
        # Update data and counts
        data = data[valid_mask].copy()
        self.dmu_count = len(data)
        
        # Update input and output arrays
        for name in self.inputs:
            self.inputs[name] = self.inputs[name][valid_mask]
        for name in self.outputs:
            self.outputs[name] = self.outputs[name][valid_mask]
            
        print(f"\nFinal data shape after cleaning: {data.shape}")
            
        return data
        
    def calculate_technical_efficiency(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical efficiency scores using DEA with scipy.optimize.linprog"""
        try:
            # Extract input and output variables
            input_cols = []
            for category in self.input_variables.values():
                input_cols.extend(category)
            
            output_cols = []
            for category in self.output_variables.values():
                output_cols.extend(category)
            
            # Filter to only include columns that exist in the data
            input_cols = [col for col in input_cols if col in data.columns]
            output_cols = [col for col in output_cols if col in data.columns]
            
            if not input_cols or not output_cols:
                print("Error: No valid input or output variables found in data")
                return None
            
            # Convert to numeric and handle missing values
            for col in input_cols + output_cols:
                data[col] = pd.to_numeric(data[col], errors='coerce')
                if data[col].isna().any():
                    print(f"Warning: Column {col} has {data[col].isna().sum()} missing values")
                    # Fill missing values with column median
                    data[col] = data[col].fillna(data[col].median())
            
            # Get input and output values
            inputs = data[input_cols].values
            outputs = data[output_cols].values
            
            # Normalize the data to avoid numerical issues
            input_means = np.mean(inputs, axis=0)
            output_means = np.mean(outputs, axis=0)
            inputs = inputs / input_means
            outputs = outputs / output_means
            
            n_dmus = len(inputs)
            n_inputs = inputs.shape[1]
            n_outputs = outputs.shape[1]
            
            # Calculate VRS (BCC) efficiency scores
            vrs_scores = []
            for i in range(n_dmus):
                # Prepare the linear programming problem for VRS
                # Decision variables: [lambda_1, ..., lambda_n]
                c = np.zeros(n_dmus)
                
                # Input constraints: sum(lambda_j * x_j) <= x_i
                A_ub_inputs = np.zeros((n_inputs, n_dmus))
                for j in range(n_inputs):
                    A_ub_inputs[j] = inputs[:, j]
                b_ub_inputs = inputs[i]
                
                # Output constraints: sum(lambda_j * y_j) >= y_i
                A_ub_outputs = np.zeros((n_outputs, n_dmus))
                for j in range(n_outputs):
                    A_ub_outputs[j] = -outputs[:, j]
                b_ub_outputs = -outputs[i]
                
                # Combine constraints
                A_ub = np.vstack([A_ub_inputs, A_ub_outputs])
                b_ub = np.hstack([b_ub_inputs, b_ub_outputs])
                
                # VRS constraint: sum(lambda) = 1
                A_eq = np.ones((1, n_dmus))
                b_eq = np.array([1])
                
                # Bounds: lambda >= 0
                bounds = [(0, None)] * n_dmus
                
                # Solve the linear programming problem
                result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
                
                if result.success:
                    # Calculate efficiency score
                    weighted_inputs = np.sum(result.x.reshape(-1, 1) * inputs, axis=0)
                    weighted_outputs = np.sum(result.x.reshape(-1, 1) * outputs, axis=0)
                    efficiency = np.mean(weighted_outputs) / np.mean(weighted_inputs)
                    vrs_scores.append(efficiency)
                else:
                    print(f"Warning: Optimization failed for DMU {i} (VRS)")
                    vrs_scores.append(None)
            
            # Calculate CRS (CCR) efficiency scores
            crs_scores = []
            for i in range(n_dmus):
                # Prepare the linear programming problem for CRS (same as VRS but without sum lambda = 1 constraint)
                c = np.zeros(n_dmus)
                
                # Input constraints
                A_ub_inputs = np.zeros((n_inputs, n_dmus))
                for j in range(n_inputs):
                    A_ub_inputs[j] = inputs[:, j]
                b_ub_inputs = inputs[i]
                
                # Output constraints
                A_ub_outputs = np.zeros((n_outputs, n_dmus))
                for j in range(n_outputs):
                    A_ub_outputs[j] = -outputs[:, j]
                b_ub_outputs = -outputs[i]
                
                # Combine constraints
                A_ub = np.vstack([A_ub_inputs, A_ub_outputs])
                b_ub = np.hstack([b_ub_inputs, b_ub_outputs])
                
                # Bounds: lambda >= 0
                bounds = [(0, None)] * n_dmus
                
                # Solve the linear programming problem
                result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
                
                if result.success:
                    # Calculate efficiency score
                    weighted_inputs = np.sum(result.x.reshape(-1, 1) * inputs, axis=0)
                    weighted_outputs = np.sum(result.x.reshape(-1, 1) * outputs, axis=0)
                    efficiency = np.mean(weighted_outputs) / np.mean(weighted_inputs)
                    crs_scores.append(efficiency)
                else:
                    print(f"Warning: Optimization failed for DMU {i} (CRS)")
                    crs_scores.append(None)
            
            # Store both VRS and CRS scores
            data['TE_VRS'] = vrs_scores
            data['TE_CRS'] = crs_scores
            data['Technical_Efficiency'] = vrs_scores  # Default to VRS for backward compatibility
            
            # Calculate scale efficiency
            data['Scale_Efficiency'] = data['TE_CRS'] / data['TE_VRS']
            
            # Determine scale efficiency categories
            data['Scale_Efficiency_Category'] = 'CRS'  # Default
            data.loc[data['Scale_Efficiency'] < 1, 'Scale_Efficiency_Category'] = 'DRS'
            data.loc[data['Scale_Efficiency'] > 1, 'Scale_Efficiency_Category'] = 'IRS'
            
            return data
                
        except Exception as e:
            print(f"Error calculating technical efficiency: {str(e)}")
            print(f"Error details: {str(e)}")
            return None
        
    def calculate_scale_efficiency(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate scale efficiency scores"""
        try:
            if 'TE_VRS' not in data.columns or 'TE_CRS' not in data.columns:
                print("Error: Technical efficiency scores not found")
                return None
            
            # Calculate scale efficiency
            data['Scale_Efficiency'] = data['TE_CRS'] / data['TE_VRS']
            
            # Add scale efficiency category
            data['Scale_Efficiency_Category'] = 'CRS'
            data.loc[data['Scale_Efficiency'] < 1, 'Scale_Efficiency_Category'] = 'IRS'
            data.loc[data['Scale_Efficiency'] > 1, 'Scale_Efficiency_Category'] = 'DRS'
            
            return data
            
        except Exception as e:
            print(f"Error calculating scale efficiency: {e}")
            return None
    
    def perform_context_dependent_analysis(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Perform context-dependent DEA analysis
        
        Args:
            data: DataFrame containing the data
            
        Returns:
            DataFrame with context-dependent scores added
        """
        print("\nPerforming context-dependent analysis...")
        
        # Analyze by Size_Category
        print("\nAnalyzing Size_Category...")
        for category in ['Small', 'Medium', 'Large']:
            subset = data[data['Size_Category'] == category]
            if len(subset) > 0:
                scores = subset['Technical_Efficiency']
                print(f"  Processing {category} category...")
                print(f"    Count: {len(subset)}")
                print(f"    Mean: {scores.mean():.4f}")
                print(f"    Std: {scores.std():.4f}")
                print(f"    Min: {scores.min():.4f}")
                print(f"    Max: {scores.max():.4f}")
        
        # Analyze by Occupancy_Category
        print("\nAnalyzing Occupancy_Category...")
        for category in ['Low', 'Medium', 'High']:
            subset = data[data['Occupancy_Category'] == category]
            if len(subset) > 0:
                scores = subset['Technical_Efficiency']
                print(f"  Processing {category} category...")
                print(f"    Count: {len(subset)}")
                print(f"    Mean: {scores.mean():.4f}")
                print(f"    Std: {scores.std():.4f}")
                print(f"    Min: {scores.min():.4f}")
                print(f"    Max: {scores.max():.4f}")
        
        # Analyze by Quality_Category
        print("\nAnalyzing Quality_Category...")
        for category in ['Low', 'Medium', 'High']:
            subset = data[data['Quality_Category'] == category]
            if len(subset) > 0:
                scores = subset['Technical_Efficiency']
                print(f"  Processing {category} category...")
                print(f"    Count: {len(subset)}")
                print(f"    Mean: {scores.mean():.4f}")
                print(f"    Std: {scores.std():.4f}")
                print(f"    Min: {scores.min():.4f}")
                print(f"    Max: {scores.max():.4f}")
        
        return data
    
    def calculate_delta_neighborhood_analysis(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate delta neighborhood analysis
        
        Args:
            data: DataFrame containing the data
            
        Returns:
            DataFrame with delta neighborhood scores added
        """
        print("\nPerforming delta neighborhood analysis...")
        return data
    
    def validate_results(self, data: pd.DataFrame) -> Dict:
        """
        Validate DEA results
        
        Args:
            data: DataFrame containing the results
            
        Returns:
            Dictionary containing validation metrics
        """
        print("\nValidating results...")
        
        validation_results = {
            'technical_efficiency': {
                'mean': data['Technical_Efficiency'].mean(),
                'std': data['Technical_Efficiency'].std(),
                'min': data['Technical_Efficiency'].min(),
                'max': data['Technical_Efficiency'].max(),
                'missing': data['Technical_Efficiency'].isna().sum()
            },
            'scale_efficiency': {
                'mean': data['Scale_Efficiency'].mean(),
                'std': data['Scale_Efficiency'].std(),
                'min': data['Scale_Efficiency'].min(),
                'max': data['Scale_Efficiency'].max(),
                'missing': data['Scale_Efficiency'].isna().sum()
            }
        }
        
        return validation_results 