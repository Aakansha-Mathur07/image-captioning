from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data
import time
import os
import tensorflow as tf
import numpy as np
import random
import math
import itertools
class Model(tf.keras.Model):
    def __init__(self, vocab_size):
        super(Model, self).__init__()
        self.batch_size = 100
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.hiddenlayer1 = 64
        self.vocab_size = vocab_size
        self.window_size = 20
        self.embedding_size = 200 
        self.rnn_size = 512
        
        self.filter1 = tf.Variable(tf.random.truncated_normal([5,5,3,self.hiddenlayer1], stddev=0.1, name="filter1"))
        self.b1 = tf.Variable(tf.random.truncated_normal([self.hiddenlayer1], stddev=.1, name="bias1"))
        self.dense1 = tf.keras.layers.Dense(self.rnn_size)
        self.dense2 = tf.keras.layers.Dense(self.rnn_size)
        
        self.E = tf.Variable(tf.random.truncated_normal(shape=[self.vocab_size, self.embedding_size], mean=0, stddev=0.1))
        self.lstm = tf.keras.layers.LSTM(self.rnn_size, return_sequences=True, return_state=True)
       
        self.dense3 = tf.keras.layers.Dense(self.rnn_size, activation='softmax')
        
    def call(self,captions,features):
        train_features = []
        train_captions = []
        for key in features.keys():
            train_features.append(features[key])
            train_captions.append(captions[key])
        train_features = tf.reshape(train_features, [self.batch_size,224,224,3])
        conv = tf.nn.conv2d(train_features, self.filter1, strides = [1, 2, 2, 1], padding="SAME") #(100, 112, 112, 64)
        conv = tf.nn.bias_add(conv, self.b1, data_format=None, name=None)
        conv = self.dense1(conv)
        conv = self.dense2(conv)
        
        
        embedding = tf.nn.embedding_lookup(self.E, train_captions)
        output, finalstate1, finalstate2 = self.lstm(embedding, initial_state=None)
        concat_feats = ([conv, output], 0)
        probs = self.dense3(concat_feats)
        print(probs)
        pass
    def loss(self, probs, labels):
        return tf.math.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, probs))
        pass
    def accuracy(self, logits, labels):
        pass
def train(model, captions, features):
    
    print('Reached train')

    features = dict(sorted(features.items()))
    captions = dict(sorted(captions.items()))


    for i in range(0, len(features), model.batch_size):
        image_features = dict(list(features.items())[i:i+model.batch_size])
        image_captions = dict(list(captions.items())[i:i+model.batch_size])
        with tf.GradientTape() as tape:
                probs = model.call(image_captions, image_features) 
               # loss = model.loss(probs, final_labels)
    pass
def test(model, test_inputs, test_labels):
    pass

def main():
    #This returns a dictionary. The image_captions is a dictionary {image_name, captions}
    #The features is a dictionary {image_name, pixels}
    #The image shape is 224,224,3
    #train: 6472
    #test: 1619
    #Vocabulary length: 5561
    train_directory = r'dataset/train/Images'
    train_file_path = r'dataset/train_captions.txt'
    train_captions, train_features =  get_data(train_directory, train_file_path)
    test_directory = r'dataset/test/Images'
    test_file_path = r'dataset/test_captions.txt'
    test_captions, test_features =  get_data(test_directory, test_file_path) 
    print(len(train_features))
    print("Completed preprocessing")
    vocabulary = set()
    for key in train_captions.keys():
        [vocabulary.update(d) for d in train_captions[key]]
    for key in test_captions.keys():
        [vocabulary.update(d) for d in test_captions[key]]
    print(len(vocabulary))
    model = Model(len(vocabulary))
    train(model, train_captions, train_features)
    return
if __name__ == '__main__':
    main()