import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize Mediapipe Hand Model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize OpenCV
cap = cv2.VideoCapture(0)

# Directory to save collected data
data_dir = 'gesture_images1'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

gesture_name = input("Enter gesture name: ")
gesture_dir = os.path.join(data_dir, gesture_name)
if not os.path.exists(gesture_dir):
    os.makedirs(gesture_dir)

data_count = 0
print("Press 'q' to stop data collection.")
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

        # Extract hand ROI
        h, w, _ = image.shape
        landmark_array = np.array([[lm.x * w, lm.y * h] for lm in hand_landmarks.landmark])
        x, y, w, h = cv2.boundingRect(landmark_array.astype(int))
        hand_roi = image[y:y+h, x:x+w]

        # Save hand ROI
        img_path = os.path.join(gesture_dir, f'{data_count}.png')
        cv2.imwrite(img_path, hand_roi)
        data_count += 1
        print(f'Saved image {data_count} for gesture {gesture_name}')

    cv2.imshow('Collecting Data', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to stop
        break

cap.release()
cv2.destroyAllWindows()
