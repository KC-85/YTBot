# infrastructure/backup.py

import logging
import shutil
import os
from datetime import datetime

def create_backup(file_path):
    """
    Creates a backup of the specified file.

    Parameters:
        file_path (str): The path of the file to back up.
    """
    if not os.path.exists(file_path):
        logging.warning(f"File {file_path} does not exist. Backup not created.")
        return

    # Construct backup file name based on the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    backup_file_path = f"{file_path}.{timestamp}.bak"

    try:
        shutil.copy(file_path, backup_file_path)
        logging.info(f"Backup created: {backup_file_path}")
    except Exception as e:
        logging.error(f"Error creating backup for {file_path}: {e}")

def restore_backup(backup_file_path, restore_path):
    """
    Restores a file from a backup.

    Parameters:
        backup_file_path (str): The path of the backup file.
        restore_path (str): The path where the file should be restored.
    """
    if not os.path.exists(backup_file_path):
        logging.warning(f"Backup file {backup_file_path} does not exist. Restore failed.")
        return

    try:
        shutil.copy(backup_file_path, restore_path)
        logging.info(f"File restored from backup: {backup_file_path} to {restore_path}")
    except Exception as e:
        logging.error(f"Error restoring backup from {backup_file_path}: {e}")

def periodic_backup(file_path, interval=900):
    """
    Periodically creates backups of a specified file at a set interval.

    Parameters:
        file_path (str): The path of the file to back up.
        interval (int): Time interval in seconds between backups.
    """
    import time
    while True:
        create_backup(file_path)
        time.sleep(interval)
