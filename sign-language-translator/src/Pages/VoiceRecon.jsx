import React, { useState } from 'react';
import axios from 'axios';

const VoiceRecognition = () => {
  const [inputTexts, setInputTexts] = useState([]);
  const [gestureImages, setGestureImages] = useState([]);
  const [isListening, setIsListening] = useState(false);
  const [messages, setMessages] = useState([]);

  const handleListen = () => {
    if (!('webkitSpeechRecognition' in window)) {
      alert('Your browser does not support speech recognition. Please try Chrome.');
      return;
    }

    const recognition = new window.webkitSpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    recognition.onstart = () => {
      setIsListening(true);
    };

    recognition.onend = () => {
      setIsListening(false);
    };

    recognition.onresult = async (event) => {
      let finalTranscript = '';
      for (let i = event.resultIndex; i < event.results.length; ++i) {
        if (event.results[i].isFinal) {
          finalTranscript += event.results[i][0].transcript;
        }
      }

      finalTranscript = finalTranscript.trim().replace(/[.,\/#!$%\^&\*;:{}=\-_`~()]+$/, '');

      if (finalTranscript) {
        setMessages((prevMessages) => [...prevMessages, finalTranscript]);
        setInputTexts((prevInputs) => [...prevInputs, finalTranscript]);
        await sendTextToBackend(finalTranscript);
      }
    };

    if (isListening) {
      recognition.stop();
    } else {
      recognition.start();
    }

    recognition.onend = () => {
      setIsListening(false);
    };
  };

  const sendTextToBackend = async (text) => {
    try {
      const response = await axios.post('http://localhost:5000/api/text_to_image', { text });
      const gesture = response.data.gesture;
      const imagePath = `http://localhost:5000/gesture_images/${gesture}`;
      setGestureImages((prevImages) => [...prevImages, imagePath]);
    } catch (error) {
      console.error('Error fetching gesture image:', error);
      alert('No gesture found for the provided text.');
    }
  };

  return (
    <div className="flex flex-col items-center mt-12">
      <h1 className="text-3xl font-bold mb-4">Voice to Sign Recognition</h1>
      <button
        onClick={handleListen}
        className={`py-2 px-4 rounded-lg mt-4 ${isListening ? 'bg-red-500' : 'bg-blue-500'} text-white`}
      >
        {isListening ? 'Stop Listening' : 'Start Listening'}
      </button>
      <div className="mt-4 w-2/3 p-4 bg-gray-100 rounded-lg max-h-screen overflow-y-auto">
        {messages.map((message, index) => (
          <div key={index} className="bg-green-500 text-white py-2 px-4 rounded-lg mb-2 text-right">
            {message}
          </div>
        ))}
      </div>
      {gestureImages.length > 0 && (
        <div className="mt-4 p-4 bg-yellow-100 rounded-lg">
          <h2 className="text-xl font-bold mb-2">Recognized Gestures</h2>
          {gestureImages.map((image, index) => (
            <img key={index} src={image} alt={`Recognized Gesture ${index + 1}`} className="mb-2" />
          ))}
        </div>
      )}
    </div>
  );
};

export default VoiceRecognition;
