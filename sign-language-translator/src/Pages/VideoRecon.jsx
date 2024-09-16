import React, { useState, useRef, useEffect } from 'react';
import io from 'socket.io-client';
import Chat from './Chat';  // Make sure you have this component or replace with your chat implementation

const socket = io('http://localhost:5000');

const VideoRecognition = () => {
  const [gesture, setGesture] = useState(null);
  const [messages, setMessages] = useState([]);
  const [lastGesture, setLastGesture] = useState(null);
  const [showGesture, setShowGesture] = useState(false);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  useEffect(() => {
    canvasRef.current = document.createElement('canvas');

    // Function to start the video stream
    const startVideo = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (error) {
        console.error('Error accessing webcam:', error);
      }
    };
    startVideo();

    // Function to send frames to the server
    const sendFrame = () => {
      const video = videoRef.current;
      if (!video) return;

      const canvas = canvasRef.current;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const context = canvas.getContext('2d');
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      const dataUrl = canvas.toDataURL('image/jpeg');

      socket.emit('frame', dataUrl);
    };

    const interval = setInterval(sendFrame, 1000); // Adjust the interval as needed
    return () => {
      clearInterval(interval);
      if (videoRef.current && videoRef.current.srcObject) {
        videoRef.current.srcObject.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  useEffect(() => {
    // Socket event listeners
    socket.on('prediction', data => {
      console.log('Received prediction:', data);  // Debug: Log the received data
      if (data.gesture) {
        setGesture(data.gesture);
        setMessages(prevMessages => [...prevMessages, data.gesture]);
        setLastGesture(data.gesture);
        setShowGesture(true);
      } else if (data.error) {
        console.error('Error:', data.error);
        setShowGesture(false);
      }
    });

    return () => {
      socket.off('prediction');
    };
  }, []);

  return (
    <div className="flex h-screen">
      <div className="relative w-1/2">
        <video ref={videoRef} className="w-full h-full object-cover" autoPlay playsInline></video>
        {showGesture && gesture && (
          <div className="absolute bottom-0 left-0 bg-yellow-100 p-4 rounded-lg m-4">
            <h2 className="text-xl font-bold">Predicted Gesture: {gesture}</h2>
          </div>
        )}
      </div>
      <div className="w-1/2 bg-gray-100 p-4 overflow-y-auto">
        <h1 className="text-3xl font-bold mb-4">Gesture Chat</h1>
        <Chat messages={messages} />
      </div>
    </div>
  );
};

export default VideoRecognition;
