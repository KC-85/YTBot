import logging
from ai.machine_learn import train_spam_detection_model, save_model

# Set up logging to display INFO level messages
logging.basicConfig(level=logging.INFO)

# Define the dataset path in your GitPod workspace
dataset_path = "ml_data_yt.xlsx"

def main():
    print("Starting model training...")

    # Train the model using the Excel dataset
    model = train_spam_detection_model(dataset_path, file_type='excel')
    
    # Check if the model was successfully trained
    if model:
        print("Model trained successfully. Saving model...")
        save_model(model, "spam_detection_model.pkl")
        print("Model saved as spam_detection_model.pkl")
    else:
        print("Model training failed. Check logs for more details.")

if __name__ == "__main__":
    main()
