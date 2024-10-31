# ai/spam_detection.py

import logging
import re

def detect_spam(message):
    """
    Detects if a message is likely to be spam.

    Parameters:
        message (str): The chat message text.

    Returns:
        bool: True if the message is considered spam, otherwise False.
    """
    # Example criteria for spam detection
    spam_indicators = [
        r"(.)\1{5,}",         # Repeated characters
        r"http[s]?://",       # URLs
        r"buy now|click here|free",  # Common spam phrases
        r"!!!|###|\$\$\$"     # Excessive use of symbols
    ]
    for indicator in spam_indicators:
        if re.search(indicator, message, re.IGNORECASE):
            logging.warning(f"Spam detected: {message}")
            return True
    logging.info("No spam detected.")
    return False

def blacklist_spam_words(words):
    """
    Adds words to a blacklist that should be flagged as spam.

    Parameters:
        words (list): A list of words to blacklist.
    """
    # Implement a mechanism to store and check blacklisted words
    logging.info(f"Spam words added to blacklist: {words}")
