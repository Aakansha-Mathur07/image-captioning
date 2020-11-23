from os import listdir
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
import tensorflow as tf
import os

def get_features(directory):
    features = {}
    for name in listdir(directory):
        filename = directory + '/' + name
        image = load_img(filename, target_size=(32, 32))
        image = img_to_array(image)
        input_arr = np.array([image])
        input_arr = input_arr/255.0
        image_name = name.split('.')[0]
        features[image_name] = input_arr
        
    return features