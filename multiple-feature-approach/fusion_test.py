from extractTest import get_test
from keras.preprocessing.text import Tokenizer
from numpy import array
from pickle import load
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Add
from nltk.translate.bleu_score import corpus_bleu
import numpy as np
import tensorflow as tf

def get_id(value, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == value:
            return word
    return None

def accuracy(model, image_captions, features_vgg,features_inception, tokenizer, max_length):
    actual_output = []
    predicted_output = []
    
    for image, captions in image_captions.items():     
        print(image)
        predicted_text = '*START*'
        for i in range(max_length):
            sequence = tokenizer.texts_to_sequences([predicted_text])[0]
            sequence = pad_sequences([sequence], maxlen=max_length)
            probs = model.predict([features_vgg[image],features_inception[image],sequence], verbose=1)
            probs = np.argmax(probs)
            word = get_id(probs, tokenizer)
            if word is None:
                break
            predicted_text += ' ' + word
            if word == '*STOP*':
                break
        caption = [caption.split() for caption in captions]
        actual_output.append(caption)
        predicted_output.append(predicted_text.split())
    print('BLEU: %f' % corpus_bleu(actual_output, predicted_output, weights=(1.0, 0, 0, 0)))

caption_filename = r'../dataset/test_captions.txt'

image_filename_vgg = r'vgg16_test_features.pkl'
image_filename_inception = r'iv3_test_features.pkl'

descriptions, vocabulary, features_vgg = get_test(caption_filename, image_filename_vgg)
description, vocabulary, features_inception = get_test(caption_filename, image_filename_inception)
all_desc = list() 
for key in descriptions.keys():
    [all_desc.append(d) for d in descriptions[key]]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_desc)
tokenized_value = tokenizer
max_length = max(len(d.split()) for d in all_desc)
model = tf.keras.models.load_model('model_fusion.h5') 

accuracy(model, descriptions, features_vgg, features_inception, tokenized_value, max_length)



#BLEU: 0.087601
