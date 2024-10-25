# auth/authentication.py

from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import logging
import os

# Path to your credentials file
CREDENTIALS_FILE = "/workspace/YTBot/credentials.json"
# Path to your token file for saving and reusing credentials
TOKEN_FILE = "token.json"

# Define the required YouTube API scopes
SCOPES = ["https://www.googleapis.com/auth/youtube.readonly"]

def initialize_youtube_client():
    """
    Initializes the YouTube API client using OAuth 2.0 credentials.
    
    Returns:
        youtube_client (Resource): The initialized YouTube API client or None if failed.
    """
    credentials = None

    try:
        # Check if token file exists and load credentials
        if os.path.exists(TOKEN_FILE):
            credentials = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        
        # If credentials don't exist or are expired, use InstalledAppFlow
        if not credentials or not credentials.valid:
            if credentials and credentials.expired and credentials.refresh_token:
                logging.info("Credentials expired. Refreshing...")
                credentials.refresh(Request())
            else:
                logging.info("No valid credentials available. Initiating OAuth flow...")
                flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
                # Explicitly set the port to 8080 to match your Google Cloud Console redirect URI
                credentials = flow.run_local_server(port=8080)
                save_credentials(credentials)

        # Build the YouTube API client
        youtube_client = build("youtube", "v3", credentials=credentials)
        logging.info("YouTube API client initialized successfully.")
        return youtube_client

    except Exception as e:
        logging.error(f"Error initializing YouTube client: {e}")
        return None

def save_credentials(credentials):
    """
    Saves the OAuth credentials to a file for future use.
    
    Parameters:
        credentials (Credentials): The OAuth credentials object.
    """
    try:
        with open(TOKEN_FILE, 'w') as token:
            token.write(credentials.to_json())
        logging.info("Credentials saved successfully.")
    except Exception as e:
        logging.error(f"Error saving credentials: {e}")

def refresh_oauth_token():
    """
    Refreshes the OAuth token if it's expired or about to expire.
    """
    try:
        if os.path.exists(TOKEN_FILE):
            credentials = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

            if credentials and credentials.expired and credentials.refresh_token:
                logging.info("Refreshing OAuth token...")
                credentials.refresh(Request())
                save_credentials(credentials)
                logging.info("OAuth token refreshed successfully.")
            else:
                logging.info("OAuth token is still valid. No refresh needed.")
        else:
            logging.error("Token file not found. Run the OAuth flow to generate new credentials.")

    except Exception as e:
        logging.error(f"Error refreshing OAuth token: {e}")

if __name__ == "__main__":
    # Testing the initialization
    youtube_client = initialize_youtube_client()
    if youtube_client:
        logging.info("YouTube client successfully initialized and ready for use.")
    else:
        logging.error("YouTube client initialization failed.")
