import os
import pickle
import numpy as np
from scipy.sparse import hstack
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
import gspread
import logging
import time
from ai.sentiment_analysis import analyze_sentiment
from dotenv import load_dotenv
from googleapiclient.errors import HttpError

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()

# Authentication and client initialization
SERVICE_ACCOUNT_FILE = os.getenv('SERVICE_ACCOUNT_FILE')
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']

creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
sheets_client = gspread.authorize(creds)
youtube_client = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

# Get live chat ID
def get_live_chat_id(video_id):
    try:
        request = youtube_client.videos().list(
            part="liveStreamingDetails,snippet",
            id=video_id
        )
        response = request.execute()
        if "items" in response and response['items']:
            live_chat_id = response['items'][0].get("liveStreamingDetails", {}).get("activeLiveChatId")
            if live_chat_id:
                return live_chat_id
            else:
                logging.error("No active live chat found for this video. Ensure the video is live or try another video ID.")
        else:
            logging.error("Video not found or does not have liveStreamingDetails.")
    except Exception as e:
        logging.error(f"An error occurred while fetching live chat ID: {e}")
    return None

# Load spam detection model
def load_spam_model(model_path="spam_detection_model.pkl"):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

# Detect keywords with admin filtering
def detect_keywords(comment, keywords, is_admin=False):
    admin_phrases = ["like subscribe share", "donate to cashapp"]
    if is_admin and any(phrase.lower() in comment.lower() for phrase in admin_phrases):
        return []
    return [keyword for keyword in keywords if keyword.lower() in comment.lower()]

# Load preprocessed vectorizer
def load_preprocessed_model_data():
    with open("preprocessed_data.pkl", "rb") as f:
        X_resampled, y_resampled, tfidf_vectorizer = pickle.load(f)
    return tfidf_vectorizer

tfidf_vectorizer = load_preprocessed_model_data()

# Determine if a comment is spam
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

# Fetch comments with pagination
def fetch_youtube_comments(live_chat_id, max_results=100, max_pages=50):
    comments = []
    page_token = None
    page_count = 0

    while page_count < max_pages:
        try:
            request = youtube_client.liveChatMessages().list(
                liveChatId=live_chat_id,
                part="snippet,authorDetails",
                maxResults=max_results,
                pageToken=page_token
            )
            response = request.execute()
            items = response.get('items', [])
            for item in items:
                author = item['authorDetails']['displayName']
                is_admin = item['authorDetails']['isChatModerator'] or item['authorDetails']['isChatOwner']
                # Check for 'displayMessage' key and handle cases where it's missing
                comment_text = item['snippet'].get('displayMessage')
                if comment_text:
                    comments.append((author, comment_text, is_admin))
                else:
                    logging.warning("Skipped a comment due to missing 'displayMessage' field.")
            page_token = response.get("nextPageToken")
            page_count += 1
            logging.info(f"Fetched page {page_count} with {len(items)} comments")
            if not page_token:
                break
        except HttpError as e:
            if e.resp.status == 403 and "rateLimitExceeded" in str(e):
                logging.warning("Rate limit exceeded. Waiting 15 seconds before retrying...")
                time.sleep(15)
            else:
                logging.error(f"An error occurred: {e}")
                break
        time.sleep(5)  # Adjust delay here if necessary

    return comments

# Append data to Google Sheets
def write_to_google_sheet(sheet_name, data):
    sheet = sheets_client.open(sheet_name).sheet1
    existing_data = sheet.get_all_values()
    start_row = len(existing_data) + 1
    headers = ["Author", "Comment", "Sentiment", "Keywords"]
    if start_row == 1:
        sheet.append_row(headers)
    sheet.update(values=data, range_name=f'A{start_row}')
    logging.info("Data appended to Google Sheet successfully.")

# Process comments and prepare data for appending
def process_comments(live_chat_id, sheet_name):
    model = load_spam_model()
    keywords = ["giveaway", "discount", "subscribe"]
    comments = fetch_youtube_comments(live_chat_id)
    processed_comments = []
    for author, comment, is_admin in comments:
        if not is_spam(comment, model, tfidf_vectorizer):
            sentiment = analyze_sentiment(comment)
            detected_keywords = detect_keywords(comment, keywords, is_admin=is_admin)
            processed_comments.append([author, comment, sentiment, ", ".join(detected_keywords)])

    write_to_google_sheet(sheet_name, processed_comments)

# Main function to execute the process in a loop
def main(video_id, sheet_name, interval=30):
    live_chat_id = get_live_chat_id(video_id)
    if not live_chat_id:
        logging.error("No live chat found for this video. Ensure live chat replay is enabled or try another video.")
        return
    logging.info(f"Starting comment fetch loop with live chat ID: {live_chat_id}")

    while True:
        process_comments(live_chat_id, sheet_name)
        logging.info(f"Waiting for {interval} seconds before fetching new comments...")
        time.sleep(interval)

if __name__ == "__main__":
    VIDEO_ID = "IzjmtrPJMS4"  # Replace with your actual video ID
    SHEET_NAME = "Data Spreadsheet"  # Replace with your Google Sheet's name
    main(VIDEO_ID, SHEET_NAME)
