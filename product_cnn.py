import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque

# Load the trained CNN model
model = tf.keras.models.load_model('gesture_cnn_model.h5')

# Initialize Mediapipe Hand Model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize OpenCV
cap = cv2.VideoCapture(0)

# Gesture names (should match the directory names used during training)
gesture_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'Hi']

# Set a threshold for confidence
confidence_threshold = 0.7

# Initialize a deque to store the last N predictions for majority voting
last_predictions = deque(maxlen=10)

def extract_hand_roi(image, hand_landmarks):
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
            # Extract hand ROI
            hand_roi = extract_hand_roi(image, hand_landmarks)
            if hand_roi.size == 0:
                print("Empty hand ROI detected.")
                continue

            # Debug: Display the hand ROI
            cv2.imshow('Hand ROI', hand_roi)

            hand_roi_resized = cv2.resize(hand_roi, (128, 128))
            hand_roi_array = np.array(hand_roi_resized) / 255.0
            hand_roi_array = np.expand_dims(hand_roi_array, axis=0)

            # Predict gesture
            predictions = model.predict(hand_roi_array)
            predicted_index = np.argmax(predictions)
            confidence = predictions[0][predicted_index]
            predicted_gesture = gesture_names[predicted_index] if confidence >= confidence_threshold else "unrecognized"

            # Store the prediction in the deque
            last_predictions.append(predicted_gesture)
            most_common_gesture = max(set(last_predictions), key=last_predictions.count)

            # Display predicted gesture
            cv2.putText(image, f'Predicted Gesture: {most_common_gesture}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        except Exception as e:
            print(f"Error during prediction: {e}")

    cv2.imshow('Real-time Gesture Recognition', image)
    if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to stop
        break

cap.release()
cv2.destroyAllWindows()
