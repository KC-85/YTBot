import gspread
from google.oauth2.service_account import Credentials

# Path to your downloaded service account key JSON file
SERVICE_ACCOUNT_FILE = 'google_sheets.json'

# Define the required scopes
SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']

# Authenticate with the service account key
creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)

# Connect to Google Sheets using gspread
client = gspread.authorize(creds)

# Open a Google Sheet by its title (make sure your service account has access)
sheet = client.open("Data spreadsheet").sheet1

# Example: Read data from cell A1
data = sheet.acell('A1').value
print(f"Data in A1: {data}")

# Example: Write data to cell A2
sheet.update(range_name='A2', values=[['Hello, Google Sheets!']])
