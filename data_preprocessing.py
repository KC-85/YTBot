import pandas as pd
import re
from html import unescape
import logging

# Set up logging to display INFO level messages
logging.basicConfig(level=logging.INFO)

def clean_comment(comment):
    """
    Cleans the comment text by decoding HTML entities and removing HTML tags.
    
    Parameters:
        comment (str): The raw comment string.
        
    Returns:
        str: The cleaned comment.
    """
    # Decode HTML entities
    comment = unescape(comment)
    # Remove HTML tags
    comment = re.sub(r'<.*?>', '', comment)
    return comment

def load_and_preprocess_data(file_path):
    """
    Loads the Excel dataset and preprocesses it.
    
    Parameters:
        file_path (str): Path to the Excel file.
        
    Returns:
        pd.DataFrame: The preprocessed dataset.
    """
    try:
        # Load the dataset
        logging.info(f"Loading dataset from {file_path}")
        df = pd.read_excel(file_path)
        
        # Check column names
        expected_columns = ['Name', 'Comment', 'Time', 'Likes', 'Reply Count', 'Spam']
        if not all(col in df.columns for col in expected_columns):
            logging.error("Dataset does not contain the expected columns.")
            return None
        
        # Convert the Time column to datetime format
        df['Time'] = pd.to_datetime(df['Time'], errors='coerce')

        # Clean the 'Comment' column
        df['Cleaned_Comment'] = df['Comment'].apply(clean_comment)
        
        # Create new features (e.g., text length)
        df['Comment_Length'] = df['Cleaned_Comment'].apply(len)
        
        logging.info("Data loaded and preprocessed successfully.")
        return df
    
    except Exception as e:
        logging.error(f"Error loading and preprocessing data: {e}")
        return None

def save_preprocessed_data(df, output_path='preprocessed_data.pkl'):
    """
    Saves the preprocessed DataFrame to a pickle file.
    
    Parameters:
        df (pd.DataFrame): The DataFrame to save.
        output_path (str): The path where the preprocessed data should be saved.
    """
    try:
        df.to_pickle(output_path)
        logging.info(f"Preprocessed data saved to {output_path}")
    except Exception as e:
        logging.error(f"Error saving preprocessed data: {e}")

def main():
    # Path to the Excel dataset
    dataset_path = 'machine_learn_data.xlsx'
    
    # Load and preprocess the data
    df = load_and_preprocess_data(dataset_path)
    
    # Save the preprocessed data
    if df is not None:
        save_preprocessed_data(df)

if __name__ == "__main__":
    main()
