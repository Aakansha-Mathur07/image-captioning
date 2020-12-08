from extractText import get_data
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
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
import numpy as np

def create_sequences(tokenizer, max_length, image_captions, photos, vocab_size):
    X1, X2, y = list(), list(), list()
    for image, captions in image_captions.items():
        for caption in captions:
            sequence = tokenizer.texts_to_sequences([caption])[0]
            for i in range(1, len(sequence)):
                train_caption, train_caption_label = sequence[:i], sequence[i]
                train_caption = pad_sequences([train_caption], maxlen=max_length)[0]
                train_caption_label = to_categorical([train_caption_label], num_classes=vocab_size)[0]
                X1.append(photos[key][0])
                X2.append(train_caption)
                y.append(train_caption_label)
    return array(X1), array(X2), array(y)



caption_filename = r'dataset/test_captions.txt'
image_filename = r'test_features.pkl'
descriptions, vocabulary, features = get_data(caption_filename, image_filename)

"""
all_desc = list()
for key in descriptions.keys():
    [all_desc.append(d) for d in descriptions[key]]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_desc)
tokenized_value = tokenizer
max_length = max(len(d.split()) for d in all_desc)
filename = 'model.h5'
model = load_model(filename)
evaluate_model(model, descriptions, features, tokenized_value, max_length)
"""
print('Done')