# config.py

import os
import logging
from dotenv import load_dotenv

def load_config():
    """
    Loads configuration settings for the YouTube Live Chat Bot from environment variables.
    If a .env file is present, it will also load these variables.
    """

    # Load environment variables from a .env file if it exists
    load_dotenv()

    # Retrieve the configuration settings from environment variables
    config = {
        "youtube_api_key": os.getenv("YOUTUBE_API_KEY"),
        "project_id": os.getenv("PROJECT_ID"),
        "credentials_file": os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "credentials.json"),
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        "poll_interval": int(os.getenv("POLL_INTERVAL", 5)),  # Interval for polling chat messages (seconds)
        "moderation_enabled": os.getenv("MODERATION_ENABLED", "true").lower() == "true",
        "database_url": os.getenv("DATABASE_URL", "mongodb://localhost:27017/"),
        "backup_bot_enabled": os.getenv("BACKUP_BOT_ENABLED", "false").lower() == "true"
    }

    # Log a message confirming configuration was loaded successfully
    logging.info("Configuration loaded successfully.")

    # Check for required keys and warn if any are missing
    required_keys = ["youtube_api_key", "project_id", "credentials_file"]
    for key in required_keys:
        if not config.get(key):
            logging.warning(f"Missing required configuration key: {key}")

    return config
