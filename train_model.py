# train_model.py

import logging
import pickle
from ai.machine_learn import save_model

# Set up logging to display INFO level messages
logging.basicConfig(level=logging.INFO)

def load_preprocessed_data(file_path):
    """
    Loads preprocessed data from a pickle file.

    Parameters:
        file_path (str): Path to the pickle file containing the preprocessed data.

    Returns:
        tuple: A tuple containing the resampled feature matrix, labels, and the TF-IDF vectorizer.
    """
    try:
        with open(file_path, 'rb') as file:
            X_resampled, y_resampled, tfidf_vectorizer = pickle.load(file)
        logging.info(f"Preprocessed data loaded from {file_path}")
        return X_resampled, y_resampled, tfidf_vectorizer
    except Exception as e:
        logging.error(f"Error loading preprocessed data: {e}")
        return None, None, None

def train_spam_detection_model(preprocessed_data_path):
    """
    Trains a spam detection model using preprocessed data.

    Parameters:
        preprocessed_data_path (str): The path to the pickle file containing preprocessed data.

    Returns:
        model (Pipeline): The trained machine learning model pipeline with the highest accuracy.
    """
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report

    try:
        # Load the preprocessed data
        X_resampled, y_resampled, tfidf_vectorizer = load_preprocessed_data(preprocessed_data_path)
        
        if X_resampled is None or y_resampled is None:
            logging.error("Failed to load preprocessed data.")
            return None

        # Split the resampled dataset into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

        # Define a set of models to test
        models = {
            'MultinomialNB': MultinomialNB(),
            'LogisticRegression': LogisticRegression(max_iter=1000),
            'RandomForest': RandomForestClassifier(n_estimators=100),
            'SVC': SVC()
        }

        best_model = None
        best_accuracy = 0

        # Iterate through models and choose the best one based on accuracy
        for name, model in models.items():
            pipeline = Pipeline([
                ('classifier', model)
            ])
            pipeline.fit(X_train, y_train)

            # Test the model
            predictions = pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            logging.info(f"Model: {name}, Accuracy: {accuracy:.2f}")

            # Log the classification report for further evaluation
            logging.info(f"\nClassification Report for {name}:\n" + classification_report(y_test, predictions))

            # Check if this model has the best accuracy so far
            if accuracy > best_accuracy:
                best_model = pipeline
                best_accuracy = accuracy

        logging.info(f"Best model chosen with accuracy: {best_accuracy:.2f}")

        return best_model

    except Exception as e:
        logging.error(f"Error training the spam detection model: {e}")
        return None

def main():
    # Path to the preprocessed data
    preprocessed_data_path = "preprocessed_data.pkl"

    print("Starting model training...")

    # Train the model using the preprocessed data
    model = train_spam_detection_model(preprocessed_data_path)
    
    # Check if the model was successfully trained
    if model:
        print("Model trained successfully. Saving model...")
        save_model(model, "spam_detection_model.pkl")
        print("Model saved as spam_detection_model.pkl")
    else:
        print("Model training failed. Check logs for more details.")

if __name__ == "__main__":
    main()
