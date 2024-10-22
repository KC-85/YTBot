# infrastructure/logging.py

import logging
import os

def setup_logging(log_file="bot.log", log_level=logging.INFO):
    """
    Sets up logging configuration.

    Parameters:
        log_file (str): The path to the log file.
        log_level (int): The logging level (e.g., logging.INFO, logging.DEBUG).
    """
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging has been configured.")

# Example usage: setup_logging()
