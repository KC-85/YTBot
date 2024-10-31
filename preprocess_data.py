import os
import numpy as np
import pandas as pd
import pickle
import re
from google.oauth2.service_account import Credentials
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from scipy.sparse import hstack
import gspread
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Google Sheets API setup
SERVICE_ACCOUNT_FILE = os.getenv('SERVICE_ACCOUNT_FILE')
SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
sheets_client = gspread.authorize(creds)

def load_data_from_sheet(sheet_name):
    """
    Load data from Google Sheets with specific headers.
    
    Parameters:
        sheet_name (str): The name of the Google Sheet.
        
    Returns:
        DataFrame: Data loaded from the Google Sheet.
    """
    sheet = sheets_client.open(sheet_name).sheet1
    # Define the headers we expect in the Google Sheet
    expected_headers = ["CONTENT", "CLASS"]
    data = sheet.get_all_records(expected_headers=expected_headers)
    df = pd.DataFrame(data)
    return df

def preprocess_data_from_sheet(sheet_name, save_path):
    """
    Preprocesses the data by applying TF-IDF, additional features, and SMOTE to balance the dataset.

    Parameters:
        sheet_name (str): Name of the Google Sheet containing data.
        save_path (str): Path to save the processed data as a pickle file.
    """
    try:
        # Load the dataset from Google Sheets
        data = load_data_from_sheet(sheet_name)

        # Ensure the necessary columns exist in the dataset
        if 'CONTENT' not in data.columns or 'CLASS' not in data.columns:
            print("Dataset must contain 'CONTENT' and 'CLASS' columns.")
            return

        # Separate features (messages) and target (labels)
        X = data['CONTENT']
        y = data['CLASS']

        # Convert the text data to numerical features using TfidfVectorizer with n-grams (bigrams and trigrams)
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), max_features=5000)
        X_tfidf = tfidf_vectorizer.fit_transform(X)

        # Generate additional features
        # Message length (number of characters)
        message_length = X.apply(len).values.reshape(-1, 1)
        
        # Count of exclamation marks
        exclamation_count = X.apply(lambda x: x.count('!')).values.reshape(-1, 1)
        
        # Count of uppercase letters
        uppercase_count = X.apply(lambda x: len(re.findall(r'[A-Z]', x))).values.reshape(-1, 1)
        
        # Keyword indicators (e.g., "free", "click", "subscribe")
        keywords = ['free', 'click', 'subscribe']
        keyword_indicators = np.array([X.str.contains(keyword, case=False).astype(int).values for keyword in keywords]).T

        # Concatenate the additional features with TF-IDF vectors
        additional_features = np.hstack((message_length, exclamation_count, uppercase_count, keyword_indicators))
        X_combined = hstack((X_tfidf, additional_features))

        # Initialize SMOTE
        smote = SMOTE(random_state=42)

        # Apply SMOTE to balance the dataset
        X_resampled, y_resampled = smote.fit_resample(X_combined, y)

        # Save the TF-IDF vectorizer and the resampled data using pickle
        with open(save_path, 'wb') as f:
            pickle.dump((X_resampled, y_resampled, tfidf_vectorizer), f)

        print(f"Data preprocessed and saved at {save_path}")

    except Exception as e:
        print(f"Error during preprocessing: {e}")

if __name__ == "__main__":
    # Specify the name of the Google Sheet containing the data
    sheet_name = "Data Spreadsheet"  # Replace with the actual name of your Google Sheet
    # Path where the preprocessed data will be saved
    save_path = "preprocessed_data.pkl"

    # Run the preprocessing
    preprocess_data_from_sheet(sheet_name, save_path)
