import os
import pickle
import numpy as np
from scipy.sparse import hstack
from google.oauth2.service_account import Credentials
import gspread
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Authenticate with Google Sheets
SERVICE_ACCOUNT_FILE = os.getenv('SERVICE_ACCOUNT_FILE')
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
sheets_client = gspread.authorize(creds)

# Load the spam detection model and TF-IDF vectorizer
def load_spam_model(model_path="spam_detection_model.pkl"):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

model = load_spam_model()
with open("preprocessed_data.pkl", "rb") as f:
    _, _, tfidf_vectorizer = pickle.load(f)

# Function to predict spam for a single comment
def is_spam(comment, model, tfidf_vectorizer):
    tfidf_features = tfidf_vectorizer.transform([comment])
    message_length = np.array([[len(comment)]])
    exclamation_count = np.array([[comment.count('!')]])
    uppercase_count = np.array([[sum(1 for c in comment if c.isupper())]])
    keywords = ['giveaway', 'discount', 'subscribe']
    keyword_indicators = np.array([[int(keyword in comment.lower()) for keyword in keywords]])
    additional_features = np.hstack((message_length, exclamation_count, uppercase_count, keyword_indicators))
    comment_features = hstack([tfidf_features, additional_features])
    return model.predict(comment_features)[0] == 1

# Function to read comments from Google Sheets and check for spam
def check_spam_in_sheet(sheet_name):
    sheet = sheets_client.open(sheet_name).sheet1
    comments = sheet.col_values(2)  # Assuming comments are in column B
    spam_results = []

    # Check each comment for spam
    for comment in comments[1:]:  # Skip header row
        spam_status = "Spam" if is_spam(comment, model, tfidf_vectorizer) else "Not Spam"
        spam_results.append([spam_status])

    # Write spam results to a new column in Google Sheets
    sheet.update('E1', [["Spam Status"]] + spam_results)  # Assuming column E is available for spam status
    print("Spam detection results updated in Google Sheet successfully.")

# Run the spam check
if __name__ == "__main__":
    SHEET_NAME = "Data Spreadsheet"  # Replace with your Google Sheet's name
    check_spam_in_sheet(SHEET_NAME)
