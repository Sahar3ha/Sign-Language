import base64
import numpy as np
import cv2
import tensorflow as tf
import mediapipe as mp
from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO, emit

# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Allow CORS
socketio = SocketIO(app, cors_allowed_origins="*")

# Load the trained CNN model
model = tf.keras.models.load_model('gesture_cnn_modelll.h5')

# Initialize Mediapipe Hand Model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)

# Gesture names (should match the classes used during training)
gesture_names = ['A', 'B', 'C', 'D', 'E']

def process_image(image_data):
    """
    Process the image data to recognize gestures.
    :param image_data: Base64 encoded image data
    :return: Recognized gesture or 'unrecognized'
    """
    try:
        # Convert base64 string to image
        image_data = base64.b64decode(image_data.split(",")[1])
        np_arr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if image is None:
            print("Error decoding image.")
            return "unrecognized"

        # Process the image to detect hand landmarks
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if not results.multi_hand_landmarks:
            return "unrecognized"

        hand_landmarks = results.multi_hand_landmarks[0]

        # Extract hand ROI
        h, w, _ = image.shape
        landmark_array = np.array([[lm.x * w, lm.y * h] for lm in hand_landmarks.landmark])
        x, y, w, h = cv2.boundingRect(landmark_array.astype(int))
        hand_roi = image[y:y+h, x:x+w]

        if hand_roi.size == 0:
            return "unrecognized"

        # Preprocess the hand ROI for the model
        hand_roi_resized = cv2.resize(hand_roi, (128, 128))
        hand_roi_array = np.array(hand_roi_resized) / 255.0
        hand_roi_array = np.expand_dims(hand_roi_array, axis=0)

        # Predict gesture
        predictions = model.predict(hand_roi_array)
        print("Predictions shape:", predictions.shape)  # Debug: Print the shape of predictions array
        print("Predictions:", predictions)  # Debug: Print the raw predictions array

        if predictions.size == 0:
            print("No predictions made by the model.")
            return "unrecognized"

        predicted_index = np.argmax(predictions)
        if predicted_index >= len(gesture_names):
            raise IndexError("Predicted index out of bounds.")
        confidence = predictions[0][predicted_index]
        print(f"Predicted gesture: {gesture_names[predicted_index]}, Confidence: {confidence}")

        if confidence >= 0.8:
            predicted_gesture = gesture_names[predicted_index]
        else:
            predicted_gesture = "unrecognized"

        return predicted_gesture

    except Exception as e:
        print("Error processing image:", str(e))
        return "unrecognized"

@socketio.on('frame')
def handle_frame(data):
    """
    Handle incoming frame data, process it and emit the prediction result.
    """
    try:
        gesture = process_image(data)
        print(f"Emitting gesture: {gesture}")  # Debug: Log the gesture being emitted
        emit('prediction', {"gesture": gesture}, broadcast=True)
    except Exception as e:
        print("Error handling frame:", str(e))
        emit('error', {"error": str(e)})

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000)
