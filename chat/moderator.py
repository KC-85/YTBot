# chat/moderator.py

import logging

def moderate_message(youtube_client, message):
    """
    Moderates a chat message based on its content.

    Parameters:
        youtube_client: The authenticated YouTube API client.
        message (dict): The message data received from the YouTube API.
    """
    text = message["text"]
    author_id = message["author_id"]

    # Example moderation: check for banned words
    banned_words = ["badword1", "badword2", "badword3"]  # Replace with actual banned words
    if any(banned_word in text.lower() for banned_word in banned_words):
        logging.warning(f"Message contains banned words. Author: {message['author']}")
        ban_user(youtube_client, author_id)

def ban_user(youtube_client, author_id):
    """
    Bans a user from the chat.

    Parameters:
        youtube_client: The authenticated YouTube API client.
        author_id (str): The ID of the user to ban.
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
        logging.info(f"User {author_id} has been banned.")
    except Exception as e:
        logging.error(f"Error banning user {author_id}: {e}")
