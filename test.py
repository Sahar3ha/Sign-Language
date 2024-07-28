import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load the model
knn = joblib.load('gesture_knn_model.pkl')

# Initialize Mediapipe Hand Model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize OpenCV
cap = cv2.VideoCapture(0)

def extract_landmarks(results):
    hand_landmarks = results.multi_hand_landmarks[0]
    landmarks = []
    for landmark in hand_landmarks.landmark:
        landmarks.append([landmark.x, landmark.y, landmark.z])
    return np.array(landmarks).flatten()

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue
    
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        landmarks = extract_landmarks(results)
        prediction = knn.predict([landmarks])[0]
        gesture_name = gestures[prediction]
        print(f'Predicted Gesture: {gesture_name}')
        
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    cv2.imshow('Hand Tracking', image)
    
    if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to stop
        break

cap.release()
cv2.destroyAllWindows()
