import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
import pickle
import re

def preprocess_data(file_path, save_path):
    """
    Preprocesses the data by applying TF-IDF, additional features, and SMOTE to balance the dataset.

    Parameters:
        file_path (str): Path to the input dataset file (Excel or CSV).
        save_path (str): Path to save the processed data as a pickle file.
    """
    try:
        # Load the dataset
        data = pd.read_excel(file_path) if file_path.endswith('.xlsx') else pd.read_csv(file_path)

        # Ensure the necessary columns exist in the dataset
        if 'CONTENT' not in data.columns or 'CLASS' not in data.columns:
            print("Dataset must contain 'CONTENT' and 'CLASS' columns.")
            return

        # Separate features (messages) and target (labels)
        X = data['CONTENT']
        y = data['CLASS']

        # Convert the text data to numerical features using TfidfVectorizer with n-grams (bigrams and trigrams)
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), max_features=5000)
        X_tfidf = tfidf_vectorizer.fit_transform(X)

        # Generate additional features
        # Message length (number of characters)
        message_length = X.apply(len).values.reshape(-1, 1)
        
        # Count of exclamation marks
        exclamation_count = X.apply(lambda x: x.count('!')).values.reshape(-1, 1)
        
        # Count of uppercase letters
        uppercase_count = X.apply(lambda x: len(re.findall(r'[A-Z]', x))).values.reshape(-1, 1)
        
        # Keyword indicators (e.g., "free", "click", "subscribe")
        keywords = ['free', 'click', 'subscribe']
        keyword_indicators = np.array([X.str.contains(keyword, case=False).astype(int).values for keyword in keywords]).T

        # Concatenate the additional features with TF-IDF vectors
        from scipy.sparse import hstack
        additional_features = np.hstack((message_length, exclamation_count, uppercase_count, keyword_indicators))
        X_combined = hstack((X_tfidf, additional_features))

        # Initialize SMOTE
        smote = SMOTE(random_state=42)

        # Apply SMOTE to balance the dataset
        X_resampled, y_resampled = smote.fit_resample(X_combined, y)

        # Save the TF-IDF vectorizer and the resampled data using pickle
        with open(save_path, 'wb') as f:
            pickle.dump((X_resampled, y_resampled, tfidf_vectorizer), f)

        print(f"Data preprocessed and saved at {save_path}")

    except Exception as e:
        print(f"Error during preprocessing: {e}")

if __name__ == "__main__":
    # Path to the dataset file (modify as needed)
    dataset_path = "ml_data_yt.xlsx"
    # Path where the preprocessed data will be saved
    save_path = "preprocessed_data.pkl"

    # Run the preprocessing
    preprocess_data(dataset_path, save_path)
