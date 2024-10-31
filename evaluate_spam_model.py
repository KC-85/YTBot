import os
import gspread
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv
from spam_check import is_spam  # Import your spam detection function from spam_check.py

# Load environment variables
load_dotenv()

# Authenticate with Google Sheets
SERVICE_ACCOUNT_FILE = os.getenv('SERVICE_ACCOUNT_FILE')
SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']

creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
sheets_client = gspread.authorize(creds)

# Function to evaluate spam detection performance
def evaluate_spam_detection(sheet_name):
    sheet = sheets_client.open(sheet_name).sheet1
    records = sheet.get_all_records()  # Get all rows, including 'True Label' column
    
    y_true = []  # True labels
    y_pred = []  # Model predictions

    # Process each row in the sheet
    for row in records:
        comment = row.get('Comment')
        is_admin = row.get('IsAdmin', 'FALSE') == 'TRUE'
        true_label = row.get('True Label', 'Not Spam')  # Expected 'Spam' or 'Not Spam'

        # Get model prediction
        prediction = "Spam" if is_spam(comment, is_admin) else "Not Spam"

        y_true.append(1 if true_label == "Spam" else 0)
        y_pred.append(1 if prediction == "Spam" else 0)

    # Calculate accuracy, precision, recall, and F1 score
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Print the results
    print("Spam Detection Evaluation:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    # Optional: Write results back to the Google Sheet
    sheet.update('F1', [["Predicted Label"]] + [["Spam" if p == 1 else "Not Spam"] for p in y_pred])

# Run the evaluation
if __name__ == "__main__":
    SHEET_NAME = "Test Data Spreadsheet"  # Replace with your test Google Sheet's name
    evaluate_spam_detection(SHEET_NAME)
