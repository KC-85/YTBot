# ai/sentiment_analysis.py

import logging
from textblob import TextBlob

def analyze_sentiment(message):
    """
    Analyzes the sentiment of a message using TextBlob.

    Parameters:
        message (str): The chat message text.

    Returns:
        str: "positive", "neutral", or "negative" based on the sentiment score.
    """
    analysis = TextBlob(message)
    sentiment_score = analysis.sentiment.polarity
    if sentiment_score > 0.1:
        logging.info(f"Message sentiment: Positive ({sentiment_score})")
        return "positive"
    elif sentiment_score < -0.1:
        logging.info(f"Message sentiment: Negative ({sentiment_score})")
        return "negative"
    else:
        logging.info(f"Message sentiment: Neutral ({sentiment_score})")
        return "neutral"

def is_negative_sentiment(message):
    """
    Determines if a message has a negative sentiment.

    Parameters:
        message (str): The chat message text.

    Returns:
        bool: True if the message has negative sentiment, otherwise False.
    """
    sentiment = analyze_sentiment(message)
    return sentiment == "negative"
