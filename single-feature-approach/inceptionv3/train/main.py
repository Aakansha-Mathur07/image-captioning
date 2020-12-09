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

def call(vocab_size, max_length):
    image_input = Input(shape=(1000,))
    features1 = Dense(256, activation='relu')(image_input)
    features2 = Dense(256, activation='relu')(features1)
    text_input = Input(shape=(max_length,))
    text1 = Embedding(vocab_size, 256, mask_zero=True)(text_input)
    text2 = LSTM(256)(text1)
    concat = Add()([features2, text2])
    output = Dense(vocab_size, activation='softmax')(concat)
    model = Model(inputs=[image_input, text_input], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    print(model.summary()) 
    return model

caption_filename = r'dataset/train_captions.txt'
image_filename = r'train_features.pkl'
descriptions, vocabulary, features = get_data(caption_filename, image_filename)
all_desc = list()
for key in descriptions.keys():
    [all_desc.append(d) for d in descriptions[key]]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_desc)
tokenized_value = tokenizer
max_length = max(len(d.split()) for d in all_desc)
X1train, X2train, ytrain = create_sequences(tokenized_value, max_length, descriptions, features, len(vocabulary))
print(X1train.shape)
print(X2train.shape)
print(ytrain.shape)
model = call(len(vocabulary), max_length)
model.fit([X1train, X2train], ytrain, epochs=1, verbose=2, batch_size=1000)
model.save('inceptionv3model'+ '.h5')


#LOSS: 5.9287