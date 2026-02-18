import os
import numpy as np
import xgboost as xgb
import scipy.signal
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

# Function to load and preprocess .mat files
def load_data(folder_path, label):
    data_list = []
    labels = []

    for file in os.listdir(folder_path):
        if file.endswith(".mat"):
            file_path = os.path.join(folder_path, file)
            annots = loadmat(file_path)
            try:
                con_list = [[element for element in upperElement] for upperElement in annots['interictal']]
            except:
                con_list = [[element for element in upperElement] for upperElement in annots['ictal']]

            # Flatten the signal
            Fsignal = np.array(con_list).flatten()

            # Apply Butterworth filter
            b, a = scipy.signal.butter(3, 0.1)
            filtered = scipy.signal.filtfilt(b, a, Fsignal)

            # Standardize the signal
            scaler = StandardScaler()
            filtered = scaler.fit_transform(filtered.reshape(-1, 1)).flatten()

            data_list.append(filtered)
            labels.append(label)

    return np.array(data_list), np.array(labels)

# Paths to datasets
seizure_path = "dataset/Seziure"   # Update with the correct folder path
seizure_free_path = "dataset/Seziure-free"

# Load and preprocess data
X_seizure, y_seizure = load_data(seizure_path, label=0)
X_seizure_free, y_seizure_free = load_data(seizure_free_path, label=1)

# Combine both datasets
X = np.vstack((X_seizure, X_seizure_free))
y = np.hstack((y_seizure, y_seizure_free))

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42, stratify=y)

# Train XGBoost Classifier
xg = xgb.XGBClassifier(n_estimators=50)
xg.fit(X_train, y_train)


# Evaluate Model
y_pred = xg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model as JSON
xg.save_model("saved_model/xgboost_model.json")
print("Model saved successfully as 'saved_model/xgboost_model.json'")
