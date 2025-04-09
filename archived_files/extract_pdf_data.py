import pdfplumber
import csv
import os
import pandas as pd

# Directory containing the PDF files
pdf_directory = './'

# Output CSV file
output_csv = 'compiled_data.csv'

# List all PDF files in the directory
pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text() + '\n'
    return text

# Function to parse text and extract structured data
def parse_text_to_data(text):
    # Placeholder for parsing logic
    # This should be customized based on the structure of your PDF content
    data = []
    lines = text.split('\n')
    for line in lines:
        # Example: Split line into columns based on a delimiter
        columns = line.split(',')  # Adjust delimiter as needed
        data.append(columns)
    return data

# Function to append data from an existing CSV file
def append_csv_data(csv_file, csv_writer):
    if not os.path.exists(csv_file):
        print(f"CSV file {csv_file} not found. Skipping.")
        return
    
    df = pd.read_csv(csv_file)
    for index, row in df.iterrows():
        csv_writer.writerow(row)

# Function to append data from an Excel file
def append_excel_data(excel_file, csv_writer):
    if not os.path.exists(excel_file):
        print(f"Excel file {excel_file} not found. Skipping.")
        return
    
    df = pd.read_excel(excel_file)
    for index, row in df.iterrows():
        csv_writer.writerow(row)

# Compile data from all PDFs into a CSV file
def compile_data_to_csv(pdf_files, output_csv):
    with open(output_csv, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Process PDF files
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_directory, pdf_file)
            print(f"Processing PDF file: {pdf_file}")
            text = extract_text_from_pdf(pdf_path)
            data = parse_text_to_data(text)
            csv_writer.writerows(data)
        
        # Append data from the existing CSV file
        print("Appending data from HCAHPS-Patient-Care-Survey.csv")
        append_csv_data('HCAHPS-Patient-Care-Survey.csv', csv_writer)
        
        # Append data from the Excel file
        print("Appending data from hospital_report_2023.xlsx")
        append_excel_data('hospital_report_2023.xlsx', csv_writer)

# Run the updated compilation process
compile_data_to_csv(pdf_files, output_csv)

print(f"Data from PDFs, CSV, and Excel files have been compiled into {output_csv}") 