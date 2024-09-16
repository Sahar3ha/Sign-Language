// Chat.js
import React from 'react';

const Chat = ({ messages }) => {
  return (
    <div className="bg-gray-100 rounded-lg p-4 max-h-screen overflow-y-auto">
      {messages.map((message, index) => (
        <div key={index} className="bg-blue-500 text-white py-2 px-4 rounded-lg mb-2 text-right">
          {message}
        </div>
      ))}
    </div>
  );
};

export default Chat;
