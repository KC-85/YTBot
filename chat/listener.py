# chat/listener.py

import logging
import time
from googleapiclient.errors import HttpError
from chat.processor import process_message

def start_chat_listener(youtube_client, live_chat_id, poll_interval=5):
    """
    Starts the chat listener that monitors YouTube live chat messages.

    Parameters:
        youtube_client: The authenticated YouTube API client.
        live_chat_id (str): The ID of the live chat to monitor.
        poll_interval (int): The interval in seconds to poll the YouTube API for new messages.
    """
    logging.info("Chat listener started.")

    while True:
        try:
            # Fetch chat messages
            response = youtube_client.liveChatMessages().list(
                liveChatId=live_chat_id,
                part="snippet,authorDetails"
            ).execute()

            for item in response.get("items", []):
                message = {
                    "id": item["id"],
                    "author": item["authorDetails"]["displayName"],
                    "author_id": item["authorDetails"]["channelId"],
                    "text": item["snippet"]["displayMessage"],
                    "timestamp": item["snippet"]["publishedAt"]
                }
                process_message(youtube_client, message)

            # Wait before polling again
            time.sleep(poll_interval)

        except HttpError as e:
            logging.error(f"API error: {e}")
            time.sleep(10)  # Wait a bit longer before retrying
        except Exception as e:
            logging.error(f"Unexpected error in chat listener: {e}")
            time.sleep(5)  # Wait before retrying in case of unexpected errors
