from ai.machine_learn import load_model
import pickle
import logging
import numpy as np
import re
from scipy.sparse import hstack

# Define additional preprocessing functions
def preprocess_message(message):
    """
    Preprocesses a single message by applying TF-IDF transformation and additional features.
    
    Parameters:
        message (str): The text message to preprocess.
    
    Returns:
        scipy.sparse matrix: Combined features (TF-IDF and additional features).
    """
    # Transform text to TF-IDF features
    message_tfidf = tfidf_vectorizer.transform([message])

    # Generate additional features
    message_length = np.array([[len(message)]])  # Message length
    exclamation_count = np.array([[message.count('!')]])  # Exclamation marks
    uppercase_count = np.array([[len(re.findall(r'[A-Z]', message))]])  # Uppercase letters

    # Keyword indicators
    keywords = ['free', 'click', 'subscribe']
    keyword_indicators = np.array([[int(keyword in message.lower()) for keyword in keywords]])

    # Concatenate all features
    additional_features = np.hstack((message_length, exclamation_count, uppercase_count, keyword_indicators))
    combined_features = hstack((message_tfidf, additional_features))

    return combined_features

def main():
    # Load the saved model and vectorizer
    model = load_model("spam_detection_model.pkl")
    try:
        with open("preprocessed_data.pkl", "rb") as file:
            _, _, tfidf_vectorizer = pickle.load(file)  # Load vectorizer only
    except Exception as e:
        logging.error(f"Error loading vectorizer: {e}")
        return

    if model:
        # Define a list of test messages
        test_messages = [
            "Subscribe to my channel for free giveaways!!!",
            "Hello everyone! Have a nice day!",
            "This is a limited-time offer, click now!",
            "Check out our new products, only today!",
            "This is a regular chat message.",
            "Free entry in a contest, register now!",
            "Thank you all for your support!",
            "Win a free vacation by signing up here!",
            "Hey guys, I just uploaded a new video, check it out!",
            "Special discounts available only for today!"
        ]

        # Loop through each message and test it
        for i, message in enumerate(test_messages, start=1):
            print(f"Test Message {i}: '{message}'")
            try:
                # Preprocess the message to get combined features
                message_features = preprocess_message(message)
                # Predict spam
                is_spam = model.predict(message_features)[0]
                print(f"Is spam: {is_spam}\n")
            except Exception as e:
                logging.error(f"Error predicting spam for message '{message}': {e}")
                print(f"Is spam: False\n")

    else:
        print("Failed to load the model. Please check the model file path and try again.")

if __name__ == "__main__":
    main()
