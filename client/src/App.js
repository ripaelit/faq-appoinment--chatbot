import React, { useEffect } from 'react';
import { Widget, addResponseMessage } from 'react-chat-widget';
import 'react-chat-widget/lib/styles.css';
import socketIOClient from 'socket.io-client';

let endPoint = 'http://localhost:5000';
let socket = socketIOClient(endPoint);

const App = () => {

  useEffect(() => {
    socket.on('message', (data) => {
      console.log(data);
      addResponseMessage(data);
    });

    return () => socket.disconnect();

  }, []);

  const handleSendMessage = (newMessage) => {
    if (newMessage !== '') {
      socket.emit('message', newMessage);
    }
  };

  return (
    <div>
      <Widget
        handleNewUserMessage={handleSendMessage}
        title='Welcome'
        subtitle='How can I help you?'
      />
    </div>
  );
};

export default App;
