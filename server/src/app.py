import nltk
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer(language='english')
import pickle
import pandas as pd
import numpy as np
import string
from dotenv import load_dotenv
load_dotenv()
import os

with open(f'src/model_data.pkl', 'rb') as f:
    data = pickle.load(f)
words = data['words']
classes = data['classes']

import json
data_set = open('src/content.json')
json_data = json.load(data_set)

with open(f'src/chatbot.pkl', 'rb') as m:
    model = pickle.load(m)

def pre_process_sentence(sentence):
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

def get_bag(sentence, words):
    sentence_words = pre_process_sentence(sentence)
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1

    return(np.array(bag))

def predict(sentence):
    ERROR_THRESHOLD = 0.25
    input_data = pd.DataFrame([get_bag(sentence, words)], dtype=float, index=['input'])
    results = model.predict([input_data])[0]
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], str(r[1])))
    
    return return_list

responses={}

for intent in json_data['intents']:
    responses[intent['tag']]=intent['responses']
 
import random

from flask import Flask;
from flask_socketio import SocketIO, send

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('MY_SECRET')

socketIo = SocketIO(app, cors_allowed_origins="*")

app.debug = True
app.host = os.getenv('HOST')

@socketIo.on('message')
def handleMessage(msg):
    print(msg)
    tag = predict(msg)[0][0]
    send(random.choice(responses[tag]))
    return None

if __name__ == '__main__':
    socketIo.run(app)