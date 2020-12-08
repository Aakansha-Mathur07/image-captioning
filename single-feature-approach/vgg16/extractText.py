from os import listdir
from pickle import dump
from tensorflow.keras.models import Model
import tensorflow as tf
import os
import string
import nltk 
nltk.download('stopwords')
nltk.download('punkt')
import string 
import re 
from nltk.corpus import stopwords 
from nltk.tokenize import RegexpTokenizer
import pickle

train_captions = dict()
vocabulary = set()
data_lines = list()
all_image_id = list()
def get_data(filename, image_filename, keyword):
    file = open(filename, 'r')
    text = file.read()

    for line in text.split('\n'):
        text_data = line.split(',')
        image_id, captions = text_data[0], text_data[1]
        image_id = image_id.split('.')[0]
        all_image_id.append(image_id)
        captions = captions.lower()      
        tokenizer = RegexpTokenizer(r'\w+')
        captions = tokenizer.tokenize(captions)
        stop_words = set(stopwords.words("english")) 
        captions = [word for word in captions if word not in stop_words] 
        captions = '*START* ' + ' '.join(captions) + ' *STOP*'
        if image_id not in train_captions:
            train_captions[image_id] = list()
        train_captions[image_id].append(captions)
        
    for key in train_captions.keys():
        [vocabulary.update(d.split()) for d in train_captions[key]]
    
    for image_id, captions in train_captions.items():
        for caption in captions:
            data_lines.append(image_id + ' ' + caption)
    
    text_features = '\n'.join(data_lines)

    file = open('train_captions.txt', 'w')
    file.write(text_features)
    file.close()

    
    set(all_image_id)
    all_features = pickle.load(open(image_filename, 'rb'))
    features = {k: all_features[k] for k in all_image_id}
    
    return train_captions, vocabulary, features