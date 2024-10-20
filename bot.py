# bot.py

import logging
import time
from auth.authentication import initialize_youtube_client
from chat.listener import start_chat_listener
from infrastructure.logging import setup_logging
from infrastructure.monitoring import start_monitoring
from config import load_config

def main():
    """
    Main entry point for the YouTube Live Chat Bot.
    This function sets up configurations, logging, and necessary services,
    then starts the chat listener to monitor and process live chat messages.
    """

    # Load configuration settings (e.g., API keys, environment variables)
    config = load_config()
    logging.info("Configuration loaded successfully.")

    # Set up logging for the bot
    setup_logging()
    logging.info("Logging setup completed.")

    # Initialize the YouTube API client
    youtube_client = initialize_youtube_client(config)
    if youtube_client is None:
        logging.error("Failed to initialize YouTube client. Exiting.")
        return
    logging.info("YouTube API client initialized successfully.")

    # Start monitoring (optional: for health checks and metrics)
    start_monitoring()

    try:
        # Start the main chat listener loop
        start_chat_listener(youtube_client, config)
    except KeyboardInterrupt:
        logging.info("Bot stopped manually.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
    finally:
        logging.info("Bot has shut down.")

if __name__ == "__main__":
    main()
