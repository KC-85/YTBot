# train_model.py

import logging
import pickle
from ai.machine_learn import save_model
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

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

def tune_model(X_train, y_train, model, param_grid):
    """
    Perform Grid Search to find the best hyperparameters for the given model.
    
    Parameters:
        X_train (array-like): Training data features.
        y_train (array-like): Training data labels.
        model: The model to tune (e.g., LogisticRegression()).
        param_grid (dict): Hyperparameter grid for the model.

    Returns:
        best_model: The model with the best found hyperparameters.
    """
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    logging.info(f"Best parameters for {model.__class__.__name__}: {grid_search.best_params_}")
    return grid_search.best_estimator_

def train_spam_detection_model(preprocessed_data_path):
    """
    Trains a spam detection model using preprocessed data.

    Parameters:
        preprocessed_data_path (str): The path to the pickle file containing preprocessed data.

    Returns:
        model (Pipeline): The trained machine learning model pipeline with the highest accuracy.
    """
    try:
        # Load the preprocessed data
        X_resampled, y_resampled, tfidf_vectorizer = load_preprocessed_data(preprocessed_data_path)
        
        if X_resampled is None or y_resampled is None:
            logging.error("Failed to load preprocessed data.")
            return None

        # Split the resampled dataset into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

        # Define parameter grids for each model
        param_grids = {
            'MultinomialNB': {},
            'LogisticRegression': {'C': [0.1, 1, 10, 100], 'max_iter': [100, 200, 500]},
            'RandomForest': {
                'n_estimators': [50, 100, 150],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'SVC': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
        }

        # Define models
        models = {
            'MultinomialNB': MultinomialNB(),
            'LogisticRegression': LogisticRegression(),
            'RandomForest': RandomForestClassifier(random_state=42),
            'SVC': SVC()
        }

        best_model = None
        best_accuracy = 0

        # Tune and evaluate each model
        for name, model in models.items():
            logging.info(f"Starting Grid Search for {name}...")
            best_estimator = tune_model(X_train, y_train, model, param_grids[name])
            
            # Evaluate the tuned model
            pipeline = Pipeline([
                ('classifier', best_estimator)
            ])
            pipeline.fit(X_train, y_train)
            
            predictions = pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            logging.info(f"Model: {name}, Tuned Accuracy: {accuracy:.2f}")

            # Log the classification report for further evaluation
            logging.info(f"\nClassification Report for {name}:\n" + classification_report(y_test, predictions))

            # Check if this model has the best accuracy so far
            if accuracy > best_accuracy:
                best_model = pipeline
                best_accuracy = accuracy

        logging.info(f"Best model chosen with tuned accuracy: {best_accuracy:.2f}")

        return best_model

    except Exception as e:
        logging.error(f"Error training the spam detection model: {e}")
        return None

def main():
    # Path to the preprocessed data
    preprocessed_data_path = "preprocessed_data.pkl"

    print("Starting model training with hyperparameter tuning...")

    # Train the model using the preprocessed data and hyperparameter tuning
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
