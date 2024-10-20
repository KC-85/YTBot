# auth/secrets.py

import logging
import os
from google.cloud import secretmanager
from google.auth.exceptions import DefaultCredentialsError

def get_secret(secret_name, version="latest"):
    """
    Retrieves a secret value from Google Cloud Secret Manager.
    
    Parameters:
        secret_name (str): The name of the secret in Google Cloud Secret Manager.
        version (str): The version of the secret to retrieve (default is "latest").
    
    Returns:
        str: The secret value if found, otherwise None.
    """
    try:
        client = secretmanager.SecretManagerServiceClient()

        # Build the resource name of the secret version
        project_id = os.getenv("PROJECT_ID")
        if not project_id:
            logging.error("Environment variable PROJECT_ID is not set.")
            return None
        
        name = f"projects/{project_id}/secrets/{secret_name}/versions/{version}"
        
        # Access the secret version
        response = client.access_secret_version(request={"name": name})
        secret_value = response.payload.data.decode("UTF-8")

        logging.info(f"Successfully retrieved secret: {secret_name}")
        return secret_value

    except DefaultCredentialsError:
        logging.error("Google Cloud credentials not found. Please set up authentication.")
    except Exception as e:
        logging.error(f"Error retrieving secret '{secret_name}': {e}")
    
    return None

def get_env_or_secret(key, default=None):
    """
    Retrieves a value from environment variables or, if not available, from Google Cloud Secret Manager.
    
    Parameters:
        key (str): The key to look up in the environment variables or secret name.
        default (Any): The default value if the environment variable and secret are not found.
    
    Returns:
        str: The value found in the environment or secret manager, otherwise the default value.
    """
    # First, try to retrieve the value from environment variables
    value = os.getenv(key)
    if value:
        logging.info(f"Retrieved {key} from environment variables.")
        return value

    # If not found in environment, try retrieving from Google Cloud Secret Manager
    logging.info(f"{key} not found in environment variables. Trying Secret Manager.")
    secret_value = get_secret(key)
    if secret_value:
        return secret_value

    logging.warning(f"Using default value for {key}.")
    return default
