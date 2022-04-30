import nltk
nltk.download('punkt')
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer(language='english')
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import pandas as pd
import pickle
import random

import json
data_set = open('content.json')
json_data = json.load(data_set)

import string

words = []
classes = []
data = []
for intent in json_data['intents']:
    for pattern in intent['patterns']:
        pattern = pattern.translate(str.maketrans('', '', string.punctuation))
        w = nltk.word_tokenize(pattern)
        w = [stemmer.stem(word.lower()) for word in w]
        words.extend(w)
        data.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

training = []
output_empty = [0] * len(classes)

for item in data:
    bag = []
    item_words = item[0]
    for w in words:
        bag.append(1) if w in item_words else bag.append(0)
    
    output_row = list(output_empty)
    output_row[classes.index(item[1])] = 1
    
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:,0])
train_y = list(training[:,1])

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(np.array(train_x), np.array(train_y), epochs=250, batch_size=5, verbose=1)

def pre_process_sentence(sentence):
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

def get_bag(sentence, words, show_details=True):
    sentence_words = pre_process_sentence(sentence)
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ('found in bag: %s' % w)

    return(np.array(bag))

# save model
pickle.dump(model, open('chatbot.pkl', 'wb'))

# save data structures
pickle.dump( {'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y}, open( "chatbot-data.pkl", "wb" ) )

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

predict('what can you do')




   