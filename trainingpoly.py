import nltk
import json
from nltk.stem import WordNetLemmatizer
lmtzr = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import random

words = []
docs = []
tags = []
ignore_words = ['?', '!']

data_file = open('CS2.json').read()
json_object = json.loads(data_file)


for intent in json_object['intents']:
  for pattern in intent['patterns']:

    w = nltk.word_tokenize(pattern)
    words.extend(w)

    docs.append((w,intent['tag']))
    
    if intent['tag'] not in tags:
      tags.append(intent['tag'])

words = [lmtzr.lemmatize(w.lower()) for w in words if w not in ignore_words]

words = sorted(list(set(words)))
tags = sorted(list(set(tags)))

# print (len(documents), "documents",documents)

# print (len(tags), "classes", tags)

# print (len(words), "unique lemmatized words", words)


pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(tags, open('tags.pkl', 'wb'))


training = []
output_empty = [0] * len(tags)

for doc in docs:
    # initializing bag of words
    bag = []
    # list of tokenized words for the pattern
    word_pattern = doc[0]
    word_pattern = [lmtzr.lemmatize(word.lower()) for word in word_pattern]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in word_pattern else bag.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    tag_position = list(output_empty)
    tag_position[tags.index(doc[1])] = 1
    
    training.append([bag, tag_position])


random.shuffle(training)
training = np.array(training)
# create train and test lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")


class ChatbotModel:
    def __init__(self, model_type):
        self.model_type = model_type
    
    def create_model(self, input_shape, output_shape):
        # create model based on model_type
        if self.model_type == 'basic':
            return self.create_basic_model(input_shape, output_shape)
        elif self.model_type == 'advanced':
            return self.create_advanced_model(input_shape, output_shape)

    def create_basic_model(self, input_shape, output_shape):
        # create basic model
        model = Sequential()
        model.add(Dense(128, input_shape=input_shape, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(output_shape, activation='softmax'))
        return model

    def create_advanced_model(self, input_shape, output_shape):
        # create advanced model
        model = Sequential()
        model.add(Dense(128, input_shape=(input_shape), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(output_shape, activation='softmax'))
        return model

    def fit(self, train_x, train_y, epochs=200, batch_size=5):
        # fit model to data
        model = self.create_model(input_shape=(len(train_x[0]),), output_shape=len(train_y[0]))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=1)
        return model

model = ChatbotModel('advanced')
model.fit(train_x, train_y)
