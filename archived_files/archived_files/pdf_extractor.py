import pdfplumber
import pandas as pd
import os

def extract_hospital_data(pdf_path):
    """
    Extract hospital performance data from the PDF document.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        pd.DataFrame: DataFrame containing extracted hospital data
    """
    all_text = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Extract text from each page
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    all_text.append(text)
                    
        # Join all text
        full_text = '\n'.join(all_text)
        
        # TODO: Process the text to extract structured data
        # This will need to be customized based on the exact format of the PDF
        
        return full_text
        
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        return None

if __name__ == "__main__":
    # Adjust the path as needed
    pdf_path = "../archived_files/FFY2023-HPR-Results-by-Hospital.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"PDF file not found at: {pdf_path}")
    else:
        # Extract the first few pages as a test
        with pdfplumber.open(pdf_path) as pdf:
            # Print basic information about the PDF
            print(f"Total pages: {len(pdf.pages)}")
            
            # Extract text from first 3 pages as a sample
            for page_num in range(min(3, len(pdf.pages))):
                print(f"\n=== Page {page_num + 1} ===")
                page = pdf.pages[page_num]
                text = page.extract_text()
                print(text[:500])  # Print first 500 characters of each page 