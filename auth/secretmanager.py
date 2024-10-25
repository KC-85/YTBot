# auth/authentication.py

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import logging
import os

def initialize_youtube_client(config):
    """
    Initializes the YouTube API client using OAuth 2.0 credentials.
    
    Parameters:
        config (dict): The configuration dictionary containing API credentials and paths.
    
    Returns:
        youtube_client (Resource): The initialized YouTube API client or None if failed.
    """
    try:
        credentials_file = config.get("credentials_file", "credentials.json")
        
        # Load credentials from the credentials file
        if not os.path.exists(credentials_file):
            logging.error(f"Credentials file not found: {credentials_file}")
            return None
        
        credentials = Credentials.from_authorized_user_file(credentials_file)
        
        # Check if credentials need refreshing
        if credentials and credentials.expired and credentials.refresh_token:
            logging.info("Credentials expired. Attempting to refresh...")
            credentials.refresh(Request())
            # Save the refreshed credentials back to the file
            save_credentials(credentials, credentials_file)
            logging.info("Credentials refreshed successfully.")
        
        # Build the YouTube API client
        youtube_client = build("youtube", "v3", credentials=credentials)
        
        # Log success if the client is initialized
        if youtube_client:
            logging.info("YouTube API client initialized successfully.")
        
        return youtube_client

    except Exception as e:
        logging.error(f"Error initializing YouTube client: {e}")
        return None

def save_credentials(credentials, credentials_file):
    """
    Saves the refreshed OAuth credentials back to the credentials file.
    
    Parameters:
        credentials (Credentials): The OAuth credentials object.
        credentials_file (str): The path to the credentials file.
    """
    try:
        with open(credentials_file, "w") as token:
            token.write(credentials.to_json())
        logging.info("Credentials saved successfully.")
    except Exception as e:
        logging.error(f"Error saving credentials: {e}")

def refresh_oauth_token(config):
    """
    Refreshes the OAuth token if it's expired or about to expire.
    
    Parameters:
        config (dict): The configuration dictionary containing API credentials and paths.
    """
    try:
        credentials_file = config.get("credentials_file", "credentials.json")
        
        # Load credentials from the file
        if not os.path.exists(credentials_file):
            logging.error(f"Credentials file not found: {credentials_file}")
            return
        
        credentials = Credentials.from_authorized_user_file(credentials_file)
        
        # Check if credentials need refreshing
        if credentials and credentials.expired and credentials.refresh_token:
            logging.info("Refreshing OAuth token...")
            credentials.refresh(Request())
            save_credentials(credentials, credentials_file)
            logging.info("OAuth token refreshed successfully.")
        else:
            logging.info("OAuth token is still valid. No refresh needed.")

    except Exception as e:
        logging.error(f"Error refreshing OAuth token: {e}")
