# auth/authentication.py

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import logging
import os
import json

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
        scopes = ["https://www.googleapis.com/auth/youtube.readonly"]
        
        # Load credentials from the file
        credentials = None
        if os.path.exists(credentials_file):
            credentials = Credentials.from_authorized_user_file(credentials_file, scopes)
        else:
            logging.error(f"Credentials file not found: {credentials_file}")
            return None
        
        # Check if credentials need refreshing
        if credentials and credentials.expired and credentials.refresh_token:
            logging.info("Credentials expired. Attempting to refresh...")
            credentials.refresh(Request())
            save_credentials(credentials, credentials_file)
            logging.info("Credentials refreshed successfully.")
        elif credentials and not credentials.refresh_token:
            logging.error("No refresh token available. Re-run the OAuth flow.")
            return None
        
        # Build the YouTube API client
        youtube_client = build("youtube", "v3", credentials=credentials)
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
        scopes = ["https://www.googleapis.com/auth/youtube.readonly"]
        
        # Load credentials from the file
        if not os.path.exists(credentials_file):
            logging.error(f"Credentials file not found: {credentials_file}")
            return
        
        credentials = Credentials.from_authorized_user_file(credentials_file, scopes)
        
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

def authenticate_with_oauth_flow(config):
    """
    Authenticates with the YouTube API using OAuth flow and saves credentials.
    
    Parameters:
        config (dict): The configuration dictionary containing API credentials and paths.
    """
    try:
        client_secrets_file = config.get("client_secrets_file", "client_secret.json")
        scopes = ["https://www.googleapis.com/auth/youtube.readonly"]
        
        # Initialize the OAuth flow
        flow = InstalledAppFlow.from_client_secrets_file(client_secrets_file, scopes)
        credentials = flow.run_local_server(port=0)
        
        # Save credentials to a file for future use
        credentials_file = config.get("credentials_file", "credentials.json")
        save_credentials(credentials, credentials_file)
        logging.info("OAuth flow completed successfully and credentials saved.")
    
    except Exception as e:
        logging.error(f"Error during OAuth flow: {e}")
