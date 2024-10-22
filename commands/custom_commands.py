# commands/custom_commands.py

import logging

# Dictionary to store custom commands with their associated functions
custom_commands = {}

def add_custom_command(command, function):
    """
    Adds a custom command to the registry.

    Parameters:
        command (str): The command keyword (e.g., "!hello").
        function (callable): The function that executes the command's action.
    """
    if command in custom_commands:
        logging.warning(f"Command {command} is already registered. Overwriting.")
    custom_commands[command] = function
    logging.info(f"Custom command {command} added successfully.")

def remove_custom_command(command):
    """
    Removes a custom command from the registry.

    Parameters:
        command (str): The command keyword to remove.
    """
    if command in custom_commands:
        del custom_commands[command]
        logging.info(f"Custom command {command} removed successfully.")
    else:
        logging.warning(f"Command {command} does not exist in the registry.")

def execute_custom_command(command, *args, **kwargs):
    """
    Executes a custom command if it exists.

    Parameters:
        command (str): The command keyword to execute.
        *args, **kwargs: Additional arguments passed to the command function.

    Returns:
        bool: True if the command was executed, False otherwise.
    """
    if command in custom_commands:
        try:
            custom_commands[command](*args, **kwargs)
            logging.info(f"Custom command {command} executed successfully.")
            return True
        except Exception as e:
            logging.error(f"Error executing command {command}: {e}")
            return False
    logging.warning(f"Custom command {command} not found.")
    return False

def list_custom_commands():
    """
    Lists all currently registered custom commands.

    Returns:
        list: A list of all custom command keywords.
    """
    return list(custom_commands.keys())
