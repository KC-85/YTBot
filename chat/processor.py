# chat/processor.py

import logging
from chat.moderator import moderate_message
from commands.handler import handle_command

def process_message(youtube_client, message):
    """
    Processes a chat message received from the YouTube live chat.

    Parameters:
        youtube_client: The authenticated YouTube API client.
        message (dict): The message data received from the YouTube API.
    """
    author = message["author"]
    text = message["text"]
    logging.info(f"Processing message from {author}: {text}")

    # Check if the message is a command
    if text.startswith("!"):
        handle_command(youtube_client, message)
    else:
        # Perform moderation checks on the message
        moderate_message(youtube_client, message)
