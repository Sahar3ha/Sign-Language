import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load the trained KNN model
knn = joblib.load('gesture_knn_model.pkl')

# Initialize Mediapipe Hand Model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize OpenCV
cap = cv2.VideoCapture(0)

# Gesture names (should match the directory names used during training)
gesture_names = ['A','B','Hi']  # Replace with actual gesture names

def extract_hand_landmarks(image, hand_landmarks):
    h, w, _ = image.shape
    landmark_array = np.array([[lm.x * w, lm.y * h] for lm in hand_landmarks.landmark])
    x, y, w, h = cv2.boundingRect(landmark_array.astype(int))
    hand_roi = image[y:y+h, x:x+w]
    return hand_roi

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        try:
            # Extract landmarks
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            landmarks = np.array(landmarks).reshape(1, -1)

            # Predict gesture
            prediction = knn.predict(landmarks)
            predicted_gesture = gesture_names[prediction[0]]

            # Display predicted gesture
            cv2.putText(image, f'Predicted Gesture: {predicted_gesture}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        except Exception as e:
            print(f"Error during prediction: {e}")

    cv2.imshow('Real-time Gesture Recognition', image)

    if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to stop
        break

cap.release()
cv2.destroyAllWindows()
