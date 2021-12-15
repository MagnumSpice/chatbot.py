import io
import random
import string
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
from gtts import gTTS
import os
warnings.filterwarnings('ignore')
import speech_recognition as sr
import nltk
from nltk.stem import WordNetLemmatizer

# comment the following after first run:
# nltk.download('popular', quiet=True)
# nltk.download('nps_chat',quiet=True)
# nltk.download('punkt')
# nltk.download('wordnet')

# enough of that ^^^

posts = nltk.corpus.nps_chat.xml_posts() [:10000]

# I guess this is supposed to recognize input type as QUES.
def dialogue_act_features(post):
    features = {}
    for word in nltk.word_tokenize(post):
        features['contains({})'.format(word.lower())] = True
    return features

featuresets = [(dialogue_act_features(post.text), post.get('class')) for post in posts]
size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)

# Greeting function
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey", "yo",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "yo", "sup", "why hello there", "I am absolutely ecstatic that you are speaking to me", "hello"]
def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# Reading in input_corpus
with open('intro_join', 'r', encoding='utf8', errors ='ignore') as fin:
    raw = fin.read().lower()

# TOkenisation
# Converts to a list of sentences:
sent_tokens = nltk.sent_tokenize(raw)
# Converts to a list of words:
word_tokens = nltk.word_tokenize(raw)

















