import random
import json
from keras.models import load_model
import numpy as np
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

lmtzr = WordNetLemmatizer()

class Chatbot:
    def __init__(self, model_type):
        self.model_type = model_type
        self.model = load_model('chatbot_model.h5')
        self.words = pickle.load(open('words.pkl', 'rb'))
        self.classes = pickle.load(open('tags.pkl', 'rb'))
        self.intents_json = json.loads(open('CS2.json').read())
    
    def clean_up_sentence(self, sentence):
        sentence_words = word_tokenize(sentence)
        sentence_words = [lmtzr.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words
    
    def bag_of_words(self, sentence, show_details=True):
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(self.words)
        for s in sentence_words:
            for i, w in enumerate(self.words):
                if w == s:
                    bag[i] = 1
                    if show_details:
                        print("found in bag: %s" % w)
        return np.array(bag)
    
    def predict_class(self, sentence):
        p = self.bag_of_words(sentence, show_details=False)
        res = self.model.predict(np.array([p]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        result_list = []
        if len(results) == 0:
            result_list.append({"intent": "not_found", "probability": 0})
            return result_list
        results.sort(key=lambda x: x[1], reverse=True)
        for r in results:
            result_list.append({"intent": self.classes[r[0]], "probability": str(r[1])})
        return result_list
    
    def generate_response(self, ints):
        tag = ints[0]['intent']
        list_of_intents = self.intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
        return result
    
    def chatbot_response(self, msg):
        ints = self.predict_class(msg)
        if ints[0]['intent'] == 'not_found':
            res = "The answer to your question was not found in dictionary or Perhaps was mispelled..."
            return res
        res = self.generate_response(ints)
        return res

class MyChatbot(Chatbot):
    def __init__(self, model_type):
        super().__init__(model_type)
        self.name = 'My Chatbot'

    def greet(self):
        return "Hello, I am {}. How can I help you today?".format(self.name)
