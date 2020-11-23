from os import listdir
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
import tensorflow as tf
import os

def get_features(directory):
    features = np.zeros([8091, 32, 32, 3])
    index = 0
    for name in listdir(directory):
        filename = directory + '/' + name
        image = load_img(filename, target_size=(32, 32))
        image = img_to_array(image)
        input_arr = np.array([image])
        input_arr = input_arr/255.0
        features[index] = input_arr
        index = index + 1 
        
    return features

