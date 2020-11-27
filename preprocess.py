from os import listdir
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
import tensorflow as tf
import os
import nltk 
nltk.download('stopwords')
nltk.download('punkt')
import string 
import re 
from nltk.corpus import stopwords 
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer 
import collections
from collections import Counter

def get_data(directory, file_path):
    features = {}
    image_captions = dict()
    for name in listdir(directory):
        filename = directory + '/' + name
        image = load_img(filename, target_size=(224, 224))
        input_arr = img_to_array(image)
        input_arr = input_arr/255.0
        image_name = name.split('.')[0]
        features[image_name] = input_arr

    with open(file_path, 'r') as f:
        train_data = f.read()
    for line in train_data.split('\n'):
        line_data = line.split(',')
        image_name, captions =  line_data[0], line_data[1]
        captions = captions.lower()
        tokenizer = RegexpTokenizer(r'\w+')
        captions = tokenizer.tokenize(captions)
        stop_words = set(stopwords.words("english")) 
        captions = [word for word in captions if word not in stop_words] 
        captions = [PorterStemmer().stem(word) for word in captions]
        image_name = image_name.split('.')[0]
        if image_name in image_captions:
            image_captions[image_name].append(captions)
        else:
            image_captions[image_name] = list()
            image_captions[image_name].append(captions)

        
    return image_captions, features