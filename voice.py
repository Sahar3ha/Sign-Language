from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import signal
import threading

app = Flask(__name__)
CORS(app)

# Gesture mapping
gesture_map = {
    "hi": "Hi/1.png",
    "a": "A/1.png",
    "b": "B/1.png",
    # Add more mappings as needed
}

@app.route('/api/text_to_image', methods=['POST'])
def text_to_image():
    data = request.json
    text = data.get('text', '').lower()
    gesture_path = gesture_map.get(text)
    if gesture_path:
        print(f"Gesture path found: {gesture_path}")  # Debug print
        return jsonify({"gesture": gesture_path})
    else:
        print(f"No gesture found for text: {text}")  # Debug print
        return jsonify({"error": "No gesture found for the provided text."}), 404

@app.route('/gesture_images/<path:filename>', methods=['GET'])
def get_gesture_image(filename):
    image_path = os.path.join('gesture_images', filename)
    print(f"Fetching image from: {image_path}")  # Debug print
    if os.path.exists(image_path):
        return send_file(image_path, mimetype='image/png')
    print(f"Image not found: {image_path}")  # Debug print
    return jsonify({"error": "Image not found."}), 404

@app.route('/shutdown', methods=['POST'])
def shutdown():
    shutdown_server()
    return 'Server shutting down...'

def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()

if __name__ == "__main__":
    app.run(debug=True)
