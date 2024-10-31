# spam_check.py

import os
import pickle
import numpy as np
import re
from scipy.sparse import hstack
from google.oauth2.service_account import Credentials
import gspread
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Authenticate with Google Sheets
SERVICE_ACCOUNT_FILE = os.getenv('SERVICE_ACCOUNT_FILE')
SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
sheets_client = gspread.authorize(creds)

# Load the spam detection model and TF-IDF vectorizer
def load_spam_model(model_path="spam_detection_model.pkl"):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

# Load the model and vectorizer
model = load_spam_model()
with open("preprocessed_data.pkl", "rb") as f:
    _, _, tfidf_vectorizer = pickle.load(f)

# Main function to detect spam using both ML and rule-based detection
def is_spam(comment, is_admin=False):
    try:
        # **Machine Learning-Based Spam Detection**
        tfidf_features = tfidf_vectorizer.transform([comment])
        message_length = np.array([[len(comment)]])
        exclamation_count = np.array([[comment.count('!')]])
        uppercase_count = np.array([[sum(1 for c in comment if c.isupper())]])
        keywords = ['giveaway', 'discount', 'subscribe']
        keyword_indicators = np.array([[int(keyword in comment.lower()) for keyword in keywords]])
        additional_features = np.hstack((message_length, exclamation_count, uppercase_count, keyword_indicators))
        comment_features = hstack([tfidf_features, additional_features])

        # Check with ML model
        if model.predict(comment_features)[0] == 1:
            return True

        # **Rule-Based Spam Detection (using regex patterns)**
        spam_indicators = [
            r"(.)\1{5,}",         # Repeated characters
            r"http[s]?://",       # URLs
            r"buy now|click here|free",  # Common spam phrases
            r"!!!|###|\$\$\$"     # Excessive use of symbols
        ]

        # Skip rule-based checks for URLs if the author is an admin
        if not is_admin:
            for indicator in spam_indicators:
                if re.search(indicator, comment, re.IGNORECASE):
                    return True

    except Exception as e:
        print(f"Error in spam detection: {e}")
        return False

    return False

# Function to read comments from Google Sheets and check for spam
def check_spam_in_sheet(sheet_name):
    sheet = sheets_client.open(sheet_name).sheet1
    comments = sheet.get_all_records()  # Assuming comments are in column B, with admin status in column C
    spam_results = []

    # Check each comment for spam
    for row in comments:
        comment = row.get('Comment')
        is_admin = row.get('IsAdmin', 'FALSE') == 'TRUE'
        spam_status = "Spam" if is_spam(comment, is_admin) else "Not Spam"
        spam_results.append([spam_status])

    # Write spam results to a new column in Google Sheets
    sheet.update('E1', [["Spam Status"]] + spam_results)  # Assuming column E is available for spam status
    print("Spam detection results updated in Google Sheet successfully.")

# Run the spam check
if __name__ == "__main__":
    SHEET_NAME = "Data Spreadsheet"  # Replace with your Google Sheet's name
    check_spam_in_sheet(SHEET_NAME)
