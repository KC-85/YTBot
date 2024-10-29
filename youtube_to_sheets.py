import os
import pickle
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
import gspread
import logging
from ai.sentiment_analysis import analyze_sentiment  # Import from sentiment_analysis.py
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Authentication and client initialization
SERVICE_ACCOUNT_FILE = os.getenv('SERVICE_ACCOUNT_FILE')
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']

creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
sheets_client = gspread.authorize(creds)
youtube_client = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

def detect_keywords(comment, keywords):
    detected_keywords = [keyword for keyword in keywords if keyword.lower() in comment.lower()]
    return detected_keywords

def load_spam_model(model_path="spam_detection_model.pkl"):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def is_spam(comment, model, tfidf_vectorizer):
    comment_features = tfidf_vectorizer.transform([comment])
    return model.predict(comment_features)[0] == 1

def fetch_youtube_comments(video_id, max_results=100):
    comments = []
    request = youtube_client.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_results,
        textFormat="plainText"
    )

    response = request.execute()
    for item in response.get('items', []):
        comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
        author = item['snippet']['topLevelComment']['snippet']['authorDisplayName']
        comments.append((author, comment))
    
    return comments

def write_to_google_sheet(sheet_name, data):
    sheet = sheets_client.open(sheet_name).sheet1
    sheet.clear()
    headers = ["Author", "Comment", "Sentiment", "Keywords"]
    rows = [headers] + data
    sheet.update(range_name='A1', values=rows)
    print("Data written to Google Sheet successfully.")

def process_comments(video_id, sheet_name):
    model = load_spam_model()
    with open("preprocessed_data.pkl", "rb") as f:
        _, _, tfidf_vectorizer = pickle.load(f)

    keywords = ["giveaway", "discount", "subscribe"]
    comments = fetch_youtube_comments(video_id)
    processed_comments = []
    for author, comment in comments:
        if not is_spam(comment, model, tfidf_vectorizer):
            sentiment = analyze_sentiment(comment)  # Use the imported function
            detected_keywords = detect_keywords(comment, keywords)
            processed_comments.append([author, comment, sentiment, ", ".join(detected_keywords)])

    write_to_google_sheet(sheet_name, processed_comments)

def main(video_id, sheet_name):
    process_comments(video_id, sheet_name)

if __name__ == "__main__":
    VIDEO_ID = "Y-vWrhvRWUE"  # Replace with your actual video ID
    SHEET_NAME = "Data Spreadsheet"  # Replace with your Google Sheet's name
    main(VIDEO_ID, SHEET_NAME)
