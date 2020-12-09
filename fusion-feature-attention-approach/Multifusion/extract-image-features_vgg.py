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
    print("entered vgg")
    model = VGG16()
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    for name in listdir(directory):
        filename = directory + '/' + name
        image = load_img(filename, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        feature = model.predict(image)
        image_name = name.split('.')[0]
        features[image_name] = feature
        print('>%s' % image_name)
        
        #print(np.array(feature).shape())
        #print(feature.shape)
    return features
    
#directory =  r'dataset/train/Images'
directory = r'/Users/hilonimehta/Desktop/image-captioning-main/dataset/train/Images'
features = get_data(directory)
dump(features, open('train_features1.pkl', ' wb'))
#directory =  r'dataset/test/Images'
directory = r'/Users/hilonimehta/Desktop/image-captioning-main/dataset/test/Images'
features = get_data(directory)
dump(features, open('test_features1.pkl', 'wb'))