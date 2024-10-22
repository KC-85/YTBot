# ai/keyword_detection.py

import logging

def detect_keywords(message, keywords):
    """
    Detects if any of the specified keywords are present in the message.

    Parameters:
        message (str): The chat message text.
        keywords (list): A list of keywords to detect.

    Returns:
        list: A list of detected keywords in the message.
    """
    detected_keywords = [keyword for keyword in keywords if keyword.lower() in message.lower()]
    if detected_keywords:
        logging.info(f"Keywords detected: {detected_keywords}")
    return detected_keywords

def register_keyword_action(keyword, action):
    """
    Registers an action for a specific keyword. When the keyword is detected,
    the action (function) will be executed.

    Parameters:
        keyword (str): The keyword to watch for.
        action (callable): The function to execute when the keyword is detected.
    """
    # You can extend this function to store actions for specific keywords
    logging.info(f"Action registered for keyword: {keyword}")
