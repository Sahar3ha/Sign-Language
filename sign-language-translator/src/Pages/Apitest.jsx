import React, { useState, useEffect } from 'react';

const ApiTest = () => {
  const [message, setMessage] = useState({});

  useEffect(() => {
    const fetchMessage = async () => {
      try {
        const response = await fetch('http://localhost:5500/api/hello');
        const data = await response.json();
        setMessage(data);
      } catch (error) {
        console.error('Error fetching message:', error);
      }
    };

    fetchMessage();
  }, []);

  return (
    <div className="text-center mt-12">
      <h1 className="text-3xl font-bold mb-4">API Test</h1>
      <p className="mt-6 text-lg text-gray-600">{message.message}</p>
    </div>
  );
};

export default ApiTest;
