#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 16:52:40 2020

@author: hilonimehta
"""
from preprocess import get_data
#imports for inception-
from keras.applications.inception_v3 import InceptionV3 
from keras.applications.inception_v3 import preprocess_input
import keras.models


def inception(features):
    model = InceptionV3(weights='imagenet')
    #remove last layer of inception v3
    model_last_remove = keras.models.Model(model.inputs, model.layers[-2].output)
    
    print(len(features))
    
    inception_pred = {}
    count = 0
    for image_array in features:
        count += 1
        
        image = features[image_array]
        
        image = preprocess_input(image)
        prob = model_last_remove.predict(image)
        print(len(prob),len(prob[0]))
        inception_pred[image_array] = prob   
        if(count == 1):
            break 
        
        
        
    #print(inception_pred)
    return(inception_pred)  
    
    
    

def main():
    #This returns a dictionary. The image_captions is a dictionary {image_name, captions}
    #The features is a dictionary {image_name, pixels}
    #The image shape is 299,299,3
    
    train_directory = r'dataset/train/Images'
    train_file_path = r'dataset/train_captions.txt'
    model_type = 'inception'
    train_captions, train_features =  get_data(train_directory, train_file_path,model_type) 
    print(len(train_features))
    print(len(train_captions))
    #image_captions, features =  get_data(directory, file_path,model_type)
    #print(len(features))
    #print(len(image_captions)) 
    #print(image_captions)
    #print(features)
    test_directory = r'dataset/test/Images'
    test_file_path = r'dataset/test_captions.txt'
    test_captions, test_features =  get_data(test_directory, test_file_path, model_type=None)  
    print(len(test_features))
    print(len(test_captions))
    inception_pred = inception(train_features) 
    
    
    
    return inception_pred
    
   

if __name__ == '__main__':
    main()    