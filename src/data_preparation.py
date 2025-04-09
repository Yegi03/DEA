import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
import openpyxl
from typing import Dict
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
warnings.filterwarnings('ignore')

class DataPreparator:
    def __init__(self):
        # Define expected variables for DEA analysis
        self.input_variables = {
            'physical_resources': ['Licensed_Beds', 'Staffed_Beds'],
            'human_resources': ['Total_Surveys'],  # Using survey count as proxy for staff
            'financial_resources': ['Bed_Occupancy_Rate']  # Using occupancy as proxy for financial efficiency
        }
        
        self.output_variables = {
            'patient_care': ['Patient_Satisfaction', 'Overall_Rating'],
            'quality_metrics': ['Recommendation_Score', 'Staff_Responsiveness'],
            'operational_metrics': ['Nurse_Communication', 'Doctor_Communication'],
            'health_outcomes': ['Cleanliness', 'Quietness']
        }
        
        # Define mappings for HCAHPS data
        self.hcahps_mappings = {
            'HospitalName': 'Hospital_Name',
            'SurveysCompleted': 'Total_Surveys',
            'HospitalRatingLM': 'Patient_Satisfaction',
            'RecommendLM': 'Recommendation_Score',
            'ResponsiveLM': 'Staff_Responsiveness',
            'SummaryStar': 'Overall_Rating',
            'CommunicateNurseLM': 'Nurse_Communication',
            'CommunicateDoctorLM': 'Doctor_Communication',
            'CleanLM': 'Cleanliness',
            'QuietLM': 'Quietness'
        }
        
        # Define mappings for hospital report data
        self.hospital_report_mappings = {
            'FACILITY NAME': 'Hospital_Name',
            'LICENSED BEDS': 'Licensed_Beds',
            'BEDS SET UP AND STAFFED': 'Staffed_Beds',
            'OCCUPANCY RATE': 'Bed_Occupancy_Rate'
        }
        
        # Initialize combined_data attribute
        self.combined_data = None

    def load_hcahps_data(self, file_path):
        """Load HCAHPS survey data"""
        try:
            # Read only necessary columns to save memory
            usecols = list(self.hcahps_mappings.keys())
            df = pd.read_csv(file_path, usecols=usecols)
            print(f"\nHCAHPS data loaded. Shape: {df.shape}")
            print("Columns:", df.columns.tolist())
            return df
        except Exception as e:
            print(f"Error loading HCAHPS data: {e}")
            return None

    def load_hospital_report(self, file_path):
        """Load hospital report data from Excel"""
        try:
            # Read the Excel file
            df = pd.read_excel(file_path)
            
            # Check if file already has the expected column names
            expected_columns = ['Hospital_Name', 'Licensed_Beds', 'Staffed_Beds', 'Bed_Occupancy_Rate']
            if all(col in df.columns for col in expected_columns):
                print(f"\nHospital report data loaded. Shape: {df.shape}")
                print("Columns:", df.columns.tolist())
                print("\nFirst few rows:")
                print(df.head())
                return df
                
            # If not, try original loading method for older format files
            # Find the header row (row with 'COUNTY')
            header_row = None
            for idx, row in df.iterrows():
                if pd.notna(row[0]) and 'COUNTY' in str(row[0]).strip():
                    header_row = idx
                    break
            
            if header_row is not None:
                # Get the column names from the header row
                df.columns = df.iloc[header_row]
                df = df.iloc[header_row + 1:].reset_index(drop=True)
                
                # Remove footer rows (rows starting with 'Note:' or containing only NaN values)
                df = df[~df['COUNTY'].str.contains('Note:', na=True)]
                df = df.dropna(how='all')
                
                # Map columns to standardized names
                df = df.rename(columns=self.hospital_report_mappings)
                
                # Convert numeric columns
                numeric_cols = ['Licensed_Beds', 'Staffed_Beds', 'Bed_Occupancy_Rate']
                for col in numeric_cols:
                    if col in df.columns:
                        # Replace dots with NaN
                        df[col] = df[col].replace('.', np.nan)
                        # Convert to numeric
                        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
                
                # Select only mapped columns
                df = df[[col for col in self.hospital_report_mappings.values() if col in df.columns]]
                
                # Remove rows with missing values
                df = df.dropna()
                
                print(f"\nHospital report data loaded. Shape: {df.shape}")
                print("Columns:", df.columns.tolist())
                print("\nFirst few rows:")
                print(df.head())
                return df
            
            print("Could not find required columns in the file")
            return None
            
        except Exception as e:
            print(f"Error loading hospital report: {e}")
            return None

    def load_pdf_data(self, file_path):
        """Load extracted PDF data"""
        try:
            df = pd.read_csv(file_path, delimiter=';')
            print(f"\nPDF data loaded. Shape: {df.shape}")
            print("Columns:", df.columns.tolist())
            return df
        except Exception as e:
            print(f"Error loading PDF data: {e}")
            return None

    def clean_maternal_opioid_data(self, df):
        """Clean maternal opioid data"""
        if df is None:
            return None
        
        print("\nCleaning maternal opioid data...")
        
        try:
            # Read the file line by line
            with open('compiled_data.csv', 'r') as f:
                lines = f.readlines()
            
            # Find the start of the data
            start_idx = None
            for i, line in enumerate(lines):
                if 'Statewide' in line:
                    start_idx = i
                    break
            
            if start_idx is None:
                print("Could not find data in maternal opioid file")
                return None
            
            # Process the data
            data = []
            for line in lines[start_idx:]:
                if line.strip() and not line.startswith('Note:'):
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        county = ' '.join(parts[:-2])
                        stays = parts[-2]
                        rate = parts[-1]
                        data.append([county, stays, rate])
            
            # Create DataFrame
            maternal_df = pd.DataFrame(data, columns=['County', 'Maternal_Opioid_Stays', 'Maternal_Opioid_Rate'])
            
            # Clean the data
            maternal_df = maternal_df.replace('NR', np.nan)
            maternal_df['Maternal_Opioid_Stays'] = pd.to_numeric(maternal_df['Maternal_Opioid_Stays'].str.replace(',', ''), errors='coerce')
            maternal_df['Maternal_Opioid_Rate'] = pd.to_numeric(maternal_df['Maternal_Opioid_Rate'], errors='coerce')
            
            print(f"Processed maternal opioid data shape: {maternal_df.shape}")
            return maternal_df
            
        except Exception as e:
            print(f"Error in clean_maternal_opioid_data: {e}")
            return None

    def clean_hcahps_data(self, df):
        """Clean HCAHPS survey data"""
        if df is None:
            return None
        
        print("\nCleaning HCAHPS data...")
        
        try:
            # Clean hospital names
            df['HospitalName'] = df['HospitalName'].str.strip()
            
            # Convert ratings to numeric
            rating_cols = [col for col in df.columns if col.endswith(('LM', 'SR'))]
            for col in rating_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Replace negative values with column mean
                mean_val = df[col].mean()
                df.loc[df[col] < 0, col] = mean_val
                
                # Replace zero values with a small positive value (5% of the mean)
                df.loc[df[col] == 0, col] = mean_val * 0.05
                
                # Fill remaining NaN values with column median
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
            
            # Map columns
            df = df.rename(columns=self.hcahps_mappings)
            
            # Select only mapped columns
            df = df[[col for col in self.hcahps_mappings.values() if col in df.columns]]
            
            print(f"Processed HCAHPS data shape: {df.shape}")
            return df
            
        except Exception as e:
            print(f"Error in clean_hcahps_data: {e}")
            return None

    def clean_hospital_report(self, df):
        """Clean hospital report data"""
        if df is None:
            return None
        
        print("\nCleaning hospital report data...")
        
        try:
            # Convert numeric columns
            numeric_cols = ['Licensed_Beds', 'Staffed_Beds', 'Bed_Occupancy_Rate']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
            
            # Remove rows with missing values
            df = df.dropna()
            
            print(f"Processed hospital report shape: {df.shape}")
            return df
            
        except Exception as e:
            print(f"Error in clean_hospital_report: {e}")
            return None

    def clean_pdf_data(self, df):
        """Clean PDF extracted data"""
        if df is None:
            return None
        
        print("\nCleaning PDF data...")
        
        # Split the single column into multiple columns
        if len(df.columns) == 1:
            df = df[df.columns[0]].str.split(',', expand=True)
        
        # Clean column names
        df.columns = df.columns.str.strip().str.replace(' ', '_')
        
        return df

    def normalize_data(self, df, columns):
        """Normalize specified columns using MinMaxScaler"""
        if df is None or not columns:
            return df
        
        print("\nNormalizing data...")
        numeric_cols = [col for col in columns if col in df.columns and df[col].dtype in ['int64', 'float64']]
        
        if not numeric_cols:
            return df
            
        # Print pre-normalization statistics
        print("\nPre-normalization statistics:")
        print(df[numeric_cols].describe())
        
        # Normalize each column separately to handle different scales
        for col in numeric_cols:
            # Get column values
            values = df[col].values.reshape(-1, 1)
            
            # Add small epsilon to minimum value to avoid zero after normalization
            min_val = values.min()
            if min_val == 0:
                values = values + 0.01 * values.mean()
            
            # Normalize
            scaler = MinMaxScaler()
            df[col] = scaler.fit_transform(values)
            
            # Ensure no zero values after normalization
            df[col] = df[col].clip(lower=0.001)
        
        # Print post-normalization statistics
        print("\nPost-normalization statistics:")
        print(df[numeric_cols].describe())
        
        return df

    def map_variables(self, df, source_type):
        """Map source-specific columns to standardized DEA variables"""
        if df is None:
            return None
        
        print(f"\nMapping {source_type} variables...")
        
        # Define mappings for each source
        mappings = {
            'hcahps': self.hcahps_mappings,
            'hospital_report': self.hospital_report_mappings,
            'pdf_data': {
                # Add PDF data specific mappings
            }
        }
        
        if source_type in mappings:
            mapping = mappings[source_type]
            # Apply mappings
            df_mapped = df.rename(columns=mapping)
            print(f"- Mapped {len(mapping)} columns")
            return df_mapped
        return df

    def combine_data_sources(self, data_sources):
        """Combine data from multiple sources"""
        try:
            hcahps_data = data_sources.get('hcahps')
            hospital_report = data_sources.get('hospital_report')
            
            if hcahps_data is None or hospital_report is None:
                print("Error: Missing required data sources")
                return None
            
            # Filter HCAHPS data for Pennsylvania hospitals only
            hcahps_data = hcahps_data[hcahps_data['State'].str.contains('Pennsylvania', na=False)]
            print(f"\nNumber of PA hospitals in HCAHPS data: {len(hcahps_data)}")
            
            # Rename columns to match
            hcahps_data = hcahps_data.rename(columns=self.hcahps_mappings)
            
            # Merge the datasets on hospital name
            merged_data = pd.merge(
                hospital_report,
                hcahps_data,
                on='Hospital_Name',
                how='inner'
            )
            
            print(f"\nFinal number of hospitals after merging: {len(merged_data)}")
            
            return merged_data
            
        except Exception as e:
            print(f"Error combining data sources: {e}")
            return None

    def prepare_dea_data(self, dfs_dict):
        """Prepare data for DEA analysis"""
        print("\nPreparing data for DEA analysis...")
        
        try:
            # Clean and combine data
            hcahps_df = self.clean_hcahps_data(dfs_dict.get('hcahps'))
            hospital_df = dfs_dict.get('hospital_report')  # Already cleaned in load_hospital_report
            
            data_sources = {
                'hcahps': hcahps_df,
                'hospital_report': hospital_df
            }
            
            combined_df = self.combine_data_sources(data_sources)
            
            if combined_df is None:
                print("No data available for DEA analysis")
                return None
            
            # Get all input and output variables
            all_inputs = [var for category in self.input_variables.values() for var in category]
            all_outputs = [var for category in self.output_variables.values() for var in category]
            
            # Select only the variables we need
            dea_vars = all_inputs + all_outputs
            dea_df = combined_df[dea_vars].copy()
            
            # Save hospital names for reference
            hospital_names = combined_df['Hospital_Name'].copy()
            
            # Normalize the data
            dea_df = self.normalize_data(dea_df, dea_vars)
            
            # Add hospital names back
            dea_df['Hospital_Name'] = hospital_names
            
            # Save the prepared data
            dea_df.to_csv('prepared_data.csv', index=False)
            print(f"Prepared data saved to prepared_data.csv. Shape: {dea_df.shape}")
            
            return dea_df
            
        except Exception as e:
            print(f"Error preparing DEA data: {e}")
            return None

    def validate_data(self, df):
        """Validate the prepared data"""
        if df is None:
            return False
            
        print("\nValidating data...")
        
        try:
            # Check for missing values
            missing = df.isnull().sum()
            if missing.any():
                print("Missing values found:")
                print(missing[missing > 0])
                return False
            
            # Check for zero or negative values in inputs
            input_cols = [var for category in self.input_variables.values() for var in category]
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            
            for col in input_cols:
                if col in numeric_cols and (df[col] <= 0).any():
                    print(f"Zero or negative values found in {col}")
                    return False
            
            # Check for correlation between inputs and outputs
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            corr_matrix = df[numeric_cols].corr()
            
            print("\nCorrelation matrix:")
            print(corr_matrix)
            
            # Check for high correlations between inputs
            available_inputs = [col for col in input_cols if col in numeric_cols]
            if available_inputs:
                input_corr = corr_matrix.loc[available_inputs, available_inputs]
                high_corr = np.where(np.abs(input_corr) > 0.9)
                if len(high_corr[0]) > len(available_inputs):  # More correlations than just diagonal
                    print("\nWarning: High correlation between inputs:")
                    for i, j in zip(*high_corr):
                        if i < j:  # Only print upper triangle
                            print(f"- {available_inputs[i]} and {available_inputs[j]}: {input_corr.iloc[i,j]:.3f}")
            
            # Print final statistics
            print("\nFinal statistics:")
            print("\nInput variables:")
            for category, vars in self.input_variables.items():
                available_vars = [var for var in vars if var in numeric_cols]
                if available_vars:
                    print(f"\n{category}:")
                    print(df[available_vars].describe())
            
            print("\nOutput variables:")
            for category, vars in self.output_variables.items():
                available_vars = [var for var in vars if var in numeric_cols]
                if available_vars:
                    print(f"\n{category}:")
                    print(df[available_vars].describe())
            
            return True
            
        except Exception as e:
            print(f"Error validating data: {e}")
            return False

def main():
    # Initialize data preparator
    preparator = DataPreparator()
    
    # Load data
    hcahps_df = preparator.load_hcahps_data('HCAHPS-Patient-Care-Survey.csv')
    hospital_df = preparator.load_hospital_report('hospital_report_2023.xlsx')
    
    # Prepare data for DEA analysis
    dfs_dict = {
        'hcahps': hcahps_df,
        'hospital_report': hospital_df
    }
    
    dea_df = preparator.prepare_dea_data(dfs_dict)
    
    if dea_df is not None:
        # Validate the data
        if preparator.validate_data(dea_df):
            print("\nData preparation completed successfully!")
            
            # Print final variable statistics
            print("\nFinal data summary:")
            print("\nInput variables:")
            for category, vars in preparator.input_variables.items():
                print(f"\n{category}:")
                print(dea_df[vars].describe())
            
            print("\nOutput variables:")
            for category, vars in preparator.output_variables.items():
                print(f"\n{category}:")
                print(dea_df[vars].describe())
        else:
            print("\nData validation failed!")
    else:
        print("\nData preparation failed!")

if __name__ == "__main__":
    main() 