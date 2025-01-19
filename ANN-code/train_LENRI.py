from LENRI2 import load_and_prepare_data, build_lenri_model, train_lenri_model


# File paths to datasets
files = ["Data/features_im0.csv", "Data/features_im1.csv", "Data/features_im2.csv", "Data/features_im3.csv"]

# Load and prepare data
X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare_data(files)

# Build the LENRI model
input_shape = (X_train.shape[1],)  # Dynamically determine input shape based on features
lenri_model = build_lenri_model(input_shape)

# Train the LENRI model
save_path = "models/LENRI_model.keras"
history = train_lenri_model(lenri_model, X_train, y_train, X_val, y_val, save_path)

# Evaluate on the unbalanced test set
print("Starting Test Data")
test_loss, test_accuracy = lenri_model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")