import pandas as pd

# Create a simplified hospital report from prepared data
prepared_df = pd.read_csv('prepared_data.csv')
hospital_report = prepared_df[['Hospital_Name', 'Licensed_Beds', 'Staffed_Beds', 'Bed_Occupancy_Rate']]
hospital_report.to_excel('data/hospital_report_2023.xlsx', index=False)
print('Created hospital_report_2023.xlsx based on prepared_data.csv') 