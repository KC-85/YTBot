import pandas as pd
import logging
import re
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO)

def clean_comment(comment):
    """
    Cleans the comment by removing HTML entities and special characters.
    
    Parameters:
        comment (str): The raw comment text.
        
    Returns:
        str: The cleaned comment text.
    """
    if pd.isna(comment):
        return ""
    # Remove HTML tags and entities
    comment = re.sub(r'<[^>]*>', '', comment)  # Remove HTML tags
    comment = re.sub(r'&[^;]+;', '', comment)  # Remove HTML entities like &lt; &gt;
    # Remove special characters, keeping words and spaces
    comment = re.sub(r'[^A-Za-z0-9\s]', '', comment)
    # Convert to lowercase
    comment = comment.lower().strip()
    return comment

def load_and_preprocess_data(file_path, sheet_name='Sheet1'):
    """
    Loads the Excel dataset and preprocesses it.
    
    Parameters:
        file_path (str): Path to the Excel file.
        sheet_name (str): Name of the sheet to read from.
        
    Returns:
        pd.DataFrame: The preprocessed dataset.
    """
    try:
        # Load the dataset from a specific sheet
        logging.info(f"Loading dataset from {file_path}, sheet: {sheet_name}")
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        # Check if the required columns are present
        expected_columns = ['Name', 'Comment', 'Time', 'Likes', 'Reply Count', 'Spam']
        if not all(col in df.columns for col in expected_columns):
            logging.error("Dataset does not contain the expected columns.")
            return None
        
        # Convert the 'Time' column to datetime format
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

def save_preprocessed_data(df, output_file="preprocessed_data.pkl"):
    """
    Saves the preprocessed data to a pickle file.
    
    Parameters:
        df (pd.DataFrame): The preprocessed dataset.
        output_file (str): The path where the preprocessed data will be saved.
    """
    try:
        with open(output_file, 'wb') as file:
            pickle.dump(df, file)
        logging.info(f"Preprocessed data saved to {output_file}")
    except Exception as e:
        logging.error(f"Error saving preprocessed data: {e}")

def main():
    # Path to the Excel dataset
    dataset_path = 'machine_learn_data.xlsx'
    sheet_name = 'Sheet1'  # Change this to the actual sheet name if needed
    
    # Load and preprocess the data
    df = load_and_preprocess_data(dataset_path, sheet_name=sheet_name)
    
    # Save the preprocessed data
    if df is not None:
        save_preprocessed_data(df)

if __name__ == "__main__":
    main()
