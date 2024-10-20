# auth/__init__.py

import logging
from .authentication import initialize_youtube_client, refresh_oauth_token

def setup_authentication(config):
    """
    Sets up the authentication system for the YouTube Live Chat Bot.
    Initializes the YouTube API client and handles token refresh if necessary.
    
    Parameters:
        config (dict): The configuration dictionary containing API credentials and paths.
    
    Returns:
        youtube_client (Resource): The initialized YouTube API client.
    """
    # Initialize the YouTube API client
    youtube_client = initialize_youtube_client(config)

    if youtube_client is None:
        logging.error("Failed to initialize the YouTube client.")
        raise RuntimeError("Authentication setup failed. Please check your credentials.")

    # (Optional) Token refresh logic if needed
    refresh_oauth_token(config)

    logging.info("Authentication setup completed successfully.")
    return youtube_client
