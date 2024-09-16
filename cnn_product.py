import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# Load the trained model
model = tf.keras.models.load_model('gesture_cnn_model.h5')

# List of gesture names (adjust based on your model's training)
gesture_names = ['A', 'B', 'C', 'D']  # Update this based on your model's classes

# Initialize Mediapipe Hand Model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)

def preprocess_image(image, hand_landmarks):
    """
    Preprocess the image for model prediction.
    :param image: Input image
    :param hand_landmarks: Detected hand landmarks
    :return: Preprocessed image array
    """
    h, w, _ = image.shape
    landmark_array = np.array([[lm.x * w, lm.y * h] for lm in hand_landmarks.landmark])
    x, y, w, h = cv2.boundingRect(landmark_array.astype(int))
    hand_roi = image[y:y+h, x:x+w]

    if hand_roi.size == 0:
        return None

    hand_roi_resized = cv2.resize(hand_roi, (128, 128))
    hand_roi_array = np.array(hand_roi_resized) / 255.0
    hand_roi_array = np.expand_dims(hand_roi_array, axis=0)  # Add batch dimension
    return hand_roi_array

def predict_gesture(image, hand_landmarks):
    """
    Predict the gesture using the model.
    :param image: Input image
    :param hand_landmarks: Detected hand landmarks
    :return: Predicted gesture
    """
    preprocessed_image = preprocess_image(image, hand_landmarks)
    if preprocessed_image is None:
        return "unrecognized"

    predictions = model.predict(preprocessed_image)
    predicted_index = np.argmax(predictions)
    confidence = predictions[0][predicted_index]

    if confidence >= 0.8:  # Adjust confidence threshold as needed
        return gesture_names[predicted_index]
    else:
        return "unrecognized"

def main():
    # Open video capture
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the image to detect hand landmarks
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            gesture = predict_gesture(frame, hand_landmarks)
            
            # Display the gesture prediction on the frame
            cv2.putText(frame, f"Predicted Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Gesture Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
