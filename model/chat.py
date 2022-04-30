import nltk
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer(language='english')
import pickle
import pandas as pd
import numpy as np
import string

data = pickle.load(open( "chatbot-data.pkl", "rb" ))
words = data['words']
classes = data['classes']

import json
data_set = open('content.json')
json_data = json.load(data_set)

with open(f'chatbot.pkl', 'rb') as f:
    model = pickle.load(f)

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

def chat():
    while True:
      user_input = input('You : ')
      tag = predict(user_input)[0][0]
      print("Reply : ",random.choice(responses[tag]))
      if tag == 'goodbye':
        break
