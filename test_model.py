# test_model.py

from ai.machine_learn import load_model, predict_spam

def main():
    # Load the saved model
    model = load_model("spam_detection_model.pkl")

    if model:
        # Test with a new message
        test_message = "Subscribe to my channel for free giveaways!!!"
        is_spam = predict_spam(model, test_message)
        print(f"Message: '{test_message}'")
        print(f"Is spam: {is_spam}")

        # You can add more test cases here
        additional_test_message = "Hello everyone! Have a nice day!"
        is_spam = predict_spam(model, additional_test_message)
        print(f"Message: '{additional_test_message}'")
        print(f"Is spam: {is_spam}")
    else:
        print("Failed to load the model. Please check the model file path and try again.")

if __name__ == "__main__":
    main()
