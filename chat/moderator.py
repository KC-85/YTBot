# chat/moderator.py

import logging
from datetime import datetime
import json

# If using a database like MongoDB, uncomment the following line and import pymongo
# from pymongo import MongoClient

def moderate_message(youtube_client, message):
    """
    Moderates a chat message based on its content.

    Parameters:
        youtube_client: The authenticated YouTube API client.
        message (dict): The message data received from the YouTube API.
    """
    text = message["text"]
    author_id = message["author_id"]
    author_name = message["author"]

    # Example moderation: check for banned words
    banned_words = ["badword1", "badword2", "badword3"]  # Replace with actual banned words
    if any(banned_word in text.lower() for banned_word in banned_words):
        logging.warning(f"Message contains banned words. Author: {author_name}")
        ban_user(youtube_client, author_id, author_name, reason="Banned words detected")

def ban_user(youtube_client, author_id, author_name, reason="Violation of chat rules"):
    """
    Bans a user from the chat and records the ban in a log file or database.

    Parameters:
        youtube_client: The authenticated YouTube API client.
        author_id (str): The ID of the user to ban.
        author_name (str): The name of the user being banned.
        reason (str): The reason for banning the user.
    """
    try:
        # Example banning action
        youtube_client.liveChatBans().insert(
            part="snippet",
            body={
                "snippet": {
                    "liveChatId": "LIVE_CHAT_ID",  # Update this dynamically if needed
                    "type": "PERMANENT",
                    "bannedUserDetails": {
                        "channelId": author_id
                    }
                }
            }
        ).execute()
        logging.info(f"User {author_name} (ID: {author_id}) has been banned.")

        # Record the ban in a log file or database
        record_ban(author_id, author_name, reason)

    except Exception as e:
        logging.error(f"Error banning user {author_id}: {e}")

def record_ban(user_id, username, reason):
    """
    Records a banned user in a log file or a database.

    Parameters:
        user_id (str): The ID of the banned user.
        username (str): The display name of the banned user.
        reason (str): The reason for banning the user.
    """
    # Create a dictionary for the banned user
    ban_record = {
        "user_id": user_id,
        "username": username,
        "reason": reason,
        "timestamp": datetime.utcnow().isoformat()
    }

    # Store the ban record in a log file (banned_users.log)
    try:
        with open("banned_users.log", "a") as log_file:
            log_file.write(json.dumps(ban_record) + "\n")
        logging.info(f"Ban record for user {username} added to the log file.")
    except Exception as e:
        logging.error(f"Error recording ban for user {username}: {e}")

    # If using a database like MongoDB, store the record in the database
    # Uncomment the following code if you have a MongoDB setup
    # try:
    #     client = MongoClient("mongodb://localhost:27017/")
    #     db = client["ytbot"]
    #     bans_collection = db["banned_users"]
    #     bans_collection.insert_one(ban_record)
    #     logging.info(f"Ban record for user {username} added to the database.")
    # except Exception as e:
    #     logging.error(f"Error recording ban for user {username} in database: {e}")

def report_banned_users():
    """
    Generates a report of all banned users from the log file.

    Returns:
        List of dictionaries representing banned users.
    """
    banned_users = []
    try:
        with open("banned_users.log", "r") as log_file:
            for line in log_file:
                banned_users.append(json.loads(line.strip()))
        logging.info("Banned users report generated successfully.")
    except FileNotFoundError:
        logging.warning("No banned users log file found.")
    except Exception as e:
        logging.error(f"Error reading banned users log file: {e}")

    return banned_users
