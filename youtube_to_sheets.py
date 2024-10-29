import os
import pickle
import numpy as np
from scipy.sparse import hstack
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
import gspread
import logging
from ai.sentiment_analysis import analyze_sentiment
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

def get_video_owner_channel_id(video_id):
    request = youtube_client.videos().list(
        part="snippet",
        id=video_id
    )
    response = request.execute()
    if response['items']:
        return response['items'][0]['snippet']['channelId']
    else:
        logging.error("Failed to retrieve video owner channel ID.")
        return None

def detect_keywords(comment, keywords, is_admin=False):
    admin_phrases = ["like subscribe share", "donate to cashapp"]

    # If from an admin, skip if the comment contains typical admin phrases
    if is_admin and any(phrase.lower() in comment.lower() for phrase in admin_phrases):
        return []

    return [keyword for keyword in keywords if keyword.lower() in comment.lower()]

def load_spam_model(model_path="spam_detection_model.pkl"):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

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

def fetch_youtube_comments(video_id, max_results=100):
    comments = []
    channel_owner_id = get_video_owner_channel_id(video_id)
    
    request = youtube_client.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_results,
        textFormat="plainText"
    )

    page = 1
    while request:
        print(f"Fetching page {page} of comments...")
        response = request.execute()
        page += 1

    while request:
        response = request.execute()
        for item in response.get('items', []):
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            author = item['snippet']['topLevelComment']['snippet']['authorDisplayName']
            author_channel_id = item['snippet']['topLevelComment']['snippet']['authorChannelId']['value']
            is_admin = author_channel_id == channel_owner_id
            comments.append((author, comment, is_admin))
        
        request = youtube_client.commentThreads().list_next(request, response)

    print(f"Total comments fetched: {len(comments)}")
    return comments

def write_to_google_sheet(sheet_name, data):
    """Appends new data to a Google Sheet without clearing existing data."""
    sheet = sheets_client.open(sheet_name).sheet1
    headers = ["Author", "Comment", "Sentiment", "Keywords"]

    # Check if the sheet is empty and add headers if necessary
    if not sheet.get_all_values():
        sheet.append_row(headers)  # Add headers to an empty sheet

    # Append new rows to the sheet
    for row in data:
        sheet.append_row(row)

    print("New data appended to Google Sheet successfully.")

def process_comments(video_id, sheet_name):
    model = load_spam_model()
    with open("preprocessed_data.pkl", "rb") as f:
        _, _, tfidf_vectorizer = pickle.load(f)

    keywords = ["giveaway", "discount", "subscribe"]
    comments = fetch_youtube_comments(video_id)
    processed_comments = []
    for author, comment, is_admin in comments:
        if not is_spam(comment, model, tfidf_vectorizer):
            sentiment = analyze_sentiment(comment)
            detected_keywords = detect_keywords(comment, keywords, is_admin=is_admin)
            processed_comments.append([author, comment, sentiment, ", ".join(detected_keywords)])

    write_to_google_sheet(sheet_name, processed_comments)

def main(video_id, sheet_name):
    process_comments(video_id, sheet_name)

if __name__ == "__main__":
    VIDEO_ID = "Y-vWrhvRWUE"  # Replace with your actual video ID
    SHEET_NAME = "Data Spreadsheet"  # Replace with your Google Sheet's name
    main(VIDEO_ID, SHEET_NAME)
