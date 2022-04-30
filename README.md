# Chatbot
A simple chatbot implementation using python. Model trained with the tenserflow. Flask for backend. React for frontned.

Current implementation only has faqs feature. Appointment booking feature to be added soon.

## Server Setup

First setup a virtual environment for python in the server folder.

Then run the following command to install the required dependencies
```bash
pip install -r requirements.txt 
```

Make a .env file in the root of server folder and add the following
```env
MY_SECRET=my-secret
HOST=localhost:5000
```

Run the server with the following command
```bash
python src/app.py
```

## Client Setup

Go to client folder and install all dependencies
```bash
npm install
```

Run the client with
```bash
npm start
```