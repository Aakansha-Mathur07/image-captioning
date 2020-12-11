from os import listdir
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
import tensorflow as tf
import os
import string 
from pickle import dump
import pickle

features = dict()

def get_data(directory):
    #print(features)
    model = VGG16()
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    for name in listdir(directory):
        filename = directory + '/' + name
        image = load_img(filename, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        feature = model.predict(image)
        image_name = name.split('.')[0]
        features[image_name] = feature
        #print('>%s' % image_name)
    return features
    
directory =  r'../dataset/train/Images'
features = get_data(directory)
dump(features, open('vgg16_train_features.pkl', 'wb'))
directory =  r'../dataset/test/Images'
features = get_data(directory)
dump(features, open('vgg16_test_features.pkl', 'wb'))