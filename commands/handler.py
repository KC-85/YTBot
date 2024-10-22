# commands/handler.py

import logging
from commands.registry import command_registry, execute_command
from commands.custom_commands import execute_custom_command

def handle_command(youtube_client, message):
    """
    Handles a command received from the chat message.

    Parameters:
        youtube_client: The authenticated YouTube API client.
        message (dict): The chat message data containing the command.
    """
    text = message["text"]
    command_keyword = text.split(" ")[0]
    author = message["author"]

    logging.info(f"Handling command from {author}: {text}")

    # Check if it's a registered command
    if command_keyword in command_registry:
        execute_command(youtube_client, command_keyword, message)
    # Check if it's a custom command
    elif execute_custom_command(command_keyword, youtube_client, message):
        logging.info(f"Executed custom command: {command_keyword}")
    else:
        logging.warning(f"Unknown command: {command_keyword}")
