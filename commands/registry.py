# commands/registry.py

import logging

# Dictionary to store registered commands and their corresponding handler functions
command_registry = {}

def register_command(command, function):
    """
    Registers a command with its corresponding function.

    Parameters:
        command (str): The command keyword (e.g., "!ban").
        function (callable): The function that executes the command's action.
    """
    if command in command_registry:
        logging.warning(f"Command {command} is already registered. Overwriting.")
    command_registry[command] = function
    logging.info(f"Command {command} registered successfully.")

def execute_command(youtube_client, command, message):
    """
    Executes a registered command.

    Parameters:
        youtube_client: The authenticated YouTube API client.
        command (str): The command keyword to execute.
        message (dict): The chat message data containing the command.
    """
    if command in command_registry:
        try:
            command_registry[command](youtube_client, message)
            logging.info(f"Command {command} executed successfully.")
        except Exception as e:
            logging.error(f"Error executing command {command}: {e}")
    else:
        logging.warning(f"Command {command} not found in the registry.")

# Example commands (these should be registered when the bot starts)
def hello_command(youtube_client, message):
    author = message["author"]
    logging.info(f"Sending a hello message to {author}.")
    # Logic to send a response to the chat, e.g., "Hello, {author}!"
    # This is a placeholder example and will depend on your implementation

def ban_command(youtube_client, message):
    author_id = message["author_id"]
    logging.info(f"Banning user {author_id}.")
    # Logic to ban the user using youtube_client

# Register example commands when the module is imported
register_command("!hello", hello_command)
register_command("!ban", ban_command)
