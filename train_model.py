from ai.machine_learn import train_spam_detection_model, save_model

# Provide the full path to your Excel dataset file
dataset_path = r"ml_data_yt.xlsx"

# Train the model using the Excel file
model = train_spam_detection_model(dataset_path, file_type='excel')

# Save the model if training was successful
if model:
    save_model(model, "spam_detection_model.pkl")
