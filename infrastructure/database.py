# infrastructure/database.py

import sqlite3
import os
import logging

def get_database_connection():
    """
    Connects to the SQLite database and returns the connection.
    If the database file does not exist, it will be created automatically.

    Returns:
        sqlite3.Connection: The SQLite connection object.
    """
    db_path = os.getenv("SQLITE_DB_PATH", "ytbot.db")
    try:
        conn = sqlite3.connect(db_path)
        logging.info(f"Connected to SQLite database at {db_path}.")
        return conn
    except sqlite3.Error as e:
        logging.error(f"Error connecting to SQLite database: {e}")
        return None

def create_banned_users_table():
    """
    Creates the banned_users table if it doesn't exist.
    """
    conn = get_database_connection()
    if conn is None:
        logging.error("Failed to create the banned_users table due to database connection issues.")
        return

    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS banned_users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                username TEXT NOT NULL,
                reason TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        logging.info("banned_users table created or verified successfully.")
    except sqlite3.Error as e:
        logging.error(f"Error creating banned_users table: {e}")
    finally:
        cursor.close()
        conn.close()

def insert_banned_user(user_id, username, reason):
    """
    Inserts a banned user into the database.

    Parameters:
        user_id (str): The ID of the banned user.
        username (str): The display name of the banned user.
        reason (str): The reason for banning the user.
    """
    conn = get_database_connection()
    if conn is None:
        logging.error("Failed to insert banned user due to database connection issues.")
        return

    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO banned_users (user_id, username, reason)
            VALUES (?, ?, ?)
        """, (user_id, username, reason))
        conn.commit()
        logging.info(f"Banned user {username} (ID: {user_id}) recorded successfully.")
    except sqlite3.Error as e:
        logging.error(f"Error inserting banned user {username}: {e}")
    finally:
        cursor.close()
        conn.close()

def get_all_banned_users():
    """
    Retrieves all banned users from the database.

    Returns:
        List[tuple]: A list of tuples, each representing a banned user record.
    """
    conn = get_database_connection()
    if conn is None:
        logging.error("Failed to retrieve banned users due to database connection issues.")
        return []

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM banned_users")
        rows = cursor.fetchall()
        logging.info(f"Retrieved {len(rows)} banned users from the database.")
        return rows
    except sqlite3.Error as e:
        logging.error(f"Error retrieving banned users: {e}")
        return []
    finally:
        cursor.close()
        conn.close()

def delete_banned_user(user_id):
    """
    Deletes a banned user from the database based on user ID.

    Parameters:
        user_id (str): The ID of the user to delete.
    """
    conn = get_database_connection()
    if conn is None:
        logging.error("Failed to delete banned user due to database connection issues.")
        return

    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM banned_users WHERE user_id = ?", (user_id,))
        conn.commit()
        logging.info(f"User with ID {user_id} deleted from banned_users table.")
    except sqlite3.Error as e:
        logging.error(f"Error deleting banned user {user_id}: {e}")
    finally:
        cursor.close()
        conn.close()
