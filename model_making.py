import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Load the trained CNN model
model = tf.keras.models.load_model('gesture_cnn_model.h5')

# Initialize Mediapipe Hand Model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize OpenCV
cap = cv2.VideoCapture(0)

# Gesture names (should match the directory names used during training)
gesture_names = ['A','B','Hi']  # Replace with actual gesture names

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
            # Preprocess the image for prediction
            image_resized = cv2.resize(image, (128, 128))
            image_array = np.array(image_resized) / 255.0
            image_array = np.expand_dims(image_array, axis=0)

            # Debug: Print the shape of the image array
            print(f"Image array shape: {image_array.shape}")

            # Predict gesture
            predictions = model.predict(image_array)
            
            # Debug: Print the predictions
            print(f"Predictions: {predictions}")

            predicted_index = np.argmax(predictions)
            predicted_gesture = gesture_names[predicted_index]

            # Display predicted gesture
            cv2.putText(image, f'Predicted Gesture: {predicted_gesture}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        except Exception as e:
            print(f"Error during prediction: {e}")

    cv2.imshow('Real-time Gesture Recognition', image)

    if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to stop
        break

cap.release()
cv2.destroyAllWindows()
