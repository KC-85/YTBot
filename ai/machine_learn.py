# ai/machine_learn.py

import logging
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def train_spam_detection_model(data_path, file_type='excel'):
    """
    Trains a spam detection model using a dataset of chat messages.

    Parameters:
        data_path (str): The path to the dataset file containing labeled chat messages.
                         The file should have two columns: "CONTENT" and "CLASS".
                         "CONTENT" contains the chat text, and "CLASS" is 0 for non-spam, 1 for spam.
        file_type (str): The type of the file, either 'csv' or 'excel'.

    Returns:
        model (Pipeline): The trained machine learning model pipeline.
    """
    try:
        # Load dataset based on the file type
        if file_type == 'csv':
            data = pd.read_csv(data_path)
        elif file_type == 'excel':
            data = pd.read_excel(data_path)
        else:
            logging.error("Unsupported file type. Use 'csv' or 'excel'.")
            return None
        
        # Ensure the necessary columns exist in the dataset
        if 'CONTENT' not in data.columns or 'CLASS' not in data.columns:
            logging.error("Dataset must contain 'CONTENT' and 'CLASS' columns.")
            return None

        # Extracting the relevant columns
        messages = data['CONTENT']
        labels = data['CLASS']

        # Split dataset into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(messages, labels, test_size=0.2, random_state=42)

        # Create a pipeline for transforming the data and training the model
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english')),
            ('classifier', MultinomialNB())
        ])

        # Train the model
        pipeline.fit(X_train, y_train)
        logging.info("Model training completed.")

        # Test the model
        predictions = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        logging.info(f"Model accuracy: {accuracy:.2f}")
        logging.info("\n" + classification_report(y_test, predictions))

        return pipeline

    except Exception as e:
        logging.error(f"Error training the spam detection model: {e}")
        return None

def save_model(model, model_path="spam_detection_model.pkl"):
    """
    Saves the trained model to a file.

    Parameters:
        model (Pipeline): The trained model pipeline to save.
        model_path (str): The path where the model should be saved.
    """
    try:
        with open(model_path, 'wb') as model_file:
            pickle.dump(model, model_file)
        logging.info(f"Model saved at {model_path}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")

def load_model(model_path="spam_detection_model.pkl"):
    """
    Loads a trained model from a file.

    Parameters:
        model_path (str): The path to the model file.

    Returns:
        model (Pipeline): The loaded machine learning model pipeline.
    """
    try:
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
        logging.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None

def predict_spam(model, message):
    """
    Predicts whether a message is spam or not using the trained model.

    Parameters:
        model (Pipeline): The trained machine learning model pipeline.
        message (str): The chat message to classify.

    Returns:
        bool: True if the message is spam, otherwise False.
    """
    try:
        prediction = model.predict([message])[0]
        logging.info(f"Message: {message} | Prediction: {'Spam' if prediction == 1 else 'Not Spam'}")
        return prediction == 1
    except Exception as e:
        logging.error(f"Error predicting spam: {e}")
        return False
