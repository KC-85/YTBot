import pandas as pd

# Path to your Excel file
file_path = 'machine_learn_data.xlsx'

# Load the Excel file
excel_file = pd.ExcelFile(file_path)

# Print all sheet names
print("Sheet names:", excel_file.sheet_names)
