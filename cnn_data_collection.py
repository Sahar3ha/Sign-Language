import cv2
import os
import time
import numpy as np
import mediapipe as mp

# Initialize Mediapipe Hand Model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)

# Set up the output directory
output_dir = 'gesture_dataset'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Ensure each gesture has its own folder
gestures = ['A', 'B', 'C', 'D']  # Update with actual gesture names
for gesture in gestures:
    gesture_dir = os.path.join(output_dir, gesture)
    if not os.path.exists(gesture_dir):
        os.makedirs(gesture_dir)

def capture_images_for_gesture(gesture_name, num_images=100):
    """
    Capture images from webcam for a specific gesture.
    :param gesture_name: Name of the gesture
    :param num_images: Number of images to capture
    """
    print(f"Starting data collection for gesture: {gesture_name}")

    # Open video capture
    cap = cv2.VideoCapture(0)
    count = 0

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Process the frame to detect hand landmarks
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            h, w, _ = frame.shape
            landmark_array = np.array([[lm.x * w, lm.y * h] for lm in hand_landmarks.landmark])
            x, y, w, h = cv2.boundingRect(landmark_array.astype(int))
            hand_roi = frame[y:y+h, x:x+w]

            if hand_roi.size > 0:
                # Save the image
                file_name = os.path.join(output_dir, gesture_name, f"{gesture_name}_{int(time.time())}.jpg")
                cv2.imwrite(file_name, hand_roi)
                count += 1
                print(f"Captured image {count}/{num_images}")

        # Display the resulting frame
        cv2.imshow('Gesture Collection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    for gesture in gestures:
        capture_images_for_gesture(gesture, num_images=100)  # Adjust num_images as needed
 