# test_model.py

from ai.machine_learn import load_model, predict_spam

def main():
    # Load the saved model with the file name "spam_detection_model.pkl"
    model = load_model("spam_detection_model.pkl")

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
            is_spam = predict_spam(model, message)
            print(f"Is spam: {is_spam}\n")

    else:
        print("Failed to load the model. Please check the model file path and try again.")

if __name__ == "__main__":
    main()
