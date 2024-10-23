import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
import pickle

def preprocess_data(file_path, save_path):
    """
    Preprocesses the data by applying TF-IDF and SMOTE to balance the dataset.

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

        # Convert the text data to numerical features using TfidfVectorizer
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        X_tfidf = tfidf_vectorizer.fit_transform(X)

        # Initialize SMOTE
        smote = SMOTE(random_state=42)

        # Apply SMOTE to balance the dataset
        X_resampled, y_resampled = smote.fit_resample(X_tfidf, y)

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
