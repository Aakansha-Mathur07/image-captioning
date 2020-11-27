#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 20:55:22 2020

@author: hilonimehta
"""
import vgg_model
import inception_model
#import keras.model 
import keras.layers

vgg_pred = vgg_model.main()
print(vgg_pred)
inception_pred = inception_model.main() 
print(inception_pred)

#output = keras.layers.Dense(512, activation="relu")

for image_name in vgg_pred:
    features_vgg = vgg_pred[image_name] #flatten and give dense layer of 512
    features_inception = inception_pred[image_name] 
    print("-----")
    print(features_vgg)
    print("-----")
    print(features_inception) 
    print(type(features_vgg))
    features_vgg = features_vgg.flatten(order='C')
    features_inception = features_inception.flatten(order='C')
    output_vgg = keras.layers.Dense(512, activation="relu")(features_vgg)
    output_inception = keras.layers.Dense(512, activation="relu")(features_inception)
    output = output_vgg + output_inception
    print(output)
    print(len(output))
    print(len(output[0])) 
    break
    #model = keras.model.Model(inputs = features, )

#output = keras.model.Model()