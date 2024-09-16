import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Directory containing gesture landmarks
data_dir = 'gesture_data_npy'
gesture_names = os.listdir(data_dir)

# Extract features and labels
features = []
labels = []

for label, gesture_name in enumerate(gesture_names):
    gesture_dir = os.path.join(data_dir, gesture_name)
    for npy_name in os.listdir(gesture_dir):
        npy_path = os.path.join(gesture_dir, npy_name)
        landmarks = np.load(npy_path)
        features.append(landmarks)
        labels.append(label)

# Convert to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Check if features and labels are extracted correctly
print(f"Features shape: {features.shape}")
print(f"Labels shape: {labels.shape}")

if features.size == 0 or labels.size == 0:
    print("No data to train the model.")
else:
    # Train KNN model
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(features, labels)

    # Save the trained KNN model
    joblib.dump(knn, 'gesture_knn_model.pkl')

    print("KNN model trained and saved successfully.")
