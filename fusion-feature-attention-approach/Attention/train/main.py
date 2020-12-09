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
    print("entered create seq")
    X1, X2, y = list(), list(), list()
    for image, captions in image_captions.items():
        #print(image)
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
    image_input = Input(shape=(1000,)) #vgg input
    features1 = Dense(256, activation='relu')(image_input) #vgg + fc layer
    
    image_input_inception = Input(shape=(1000,))#inception input
    features2 = Dense(256, activation='relu')(image_input_inception)#inception + fc layer
    
    concat1 = Add()([features1,features2]) #adding vgg and inception
   
    text_input = Input(shape=(max_length,))#lstm
    text1 = Embedding(vocab_size, 256, mask_zero=True)(text_input)
    text2 = LSTM(256)(text1)#lstm 
   
    concat = Add()([concat1, text2])#Adding fusion of features to LSTM
    
    concat1 = Add()([concat,features2]) #Adding inception with fc layer again
    #concat1 = Add()([concat,image_input_inception])#Adding inception again
    
    concat2 = Add()([concat1,text2]) #Adding LSTM 
   
    output = Dense(vocab_size, activation='softmax')(concat2) #adding softmax layer
    
    model = Model(inputs=[image_input,image_input_inception, text_input], outputs=output)
  
    model.compile(loss='categorical_crossentropy', optimizer='adam')
  
    print(model.summary()) 
    return model

#caption_filename = r'../dataset/train_captions.txt'
caption_filename = r'/Users/hilonimehta/Desktop/image-captioning-main/dataset/train_captions.txt'
#image_filename = r'/Users/hilonimehta/Desktop/image-captioning-main/inbuilt_conv_2/train_features3.pkl'
image_filename = r'/Users/hilonimehta/Desktop/image-captioning-main/single-feature-approach/vgg16/train/train_features.pkl'
#image_filename = r'train_features.pkl'
descriptions, vocabulary, features = get_data(caption_filename, image_filename)
image_filename_inception = r'/Users/hilonimehta/Desktop/image-captioning-main/single-feature-approach/inceptionv3/train/train_features.pkl'
all_desc = list()
all_desc_inception = list()
descriptions_inception, vocabulary_inception, features_inception = get_data(caption_filename, image_filename_inception)
for key in descriptions.keys():
    [all_desc.append(d) for d in descriptions[key]]
    

    
'''for key in descriptions_inception.keys():
    [all_desc_inception.append(d) for d in descriptions[key]]
    
print(all_desc)
print("---------------------------------------")
print(all_desc_inception)'''#no need since description and vocabulary remains the same

if(descriptions == descriptions_inception):
    print("True")
    
if(vocabulary == vocabulary_inception):
    print("True")

   
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_desc)
tokenized_value = tokenizer
max_length = max(len(d.split()) for d in all_desc)
X1train, X2train, ytrain = create_sequences(tokenized_value, max_length, descriptions, features, len(vocabulary))
X3train, X4train, ytrain_inception = create_sequences(tokenized_value, max_length, descriptions, features_inception, len(vocabulary))
print(X1train.shape)
print(X2train.shape)
print(ytrain.shape)
print(X3train.shape)
print(X4train.shape)
print(ytrain_inception.shape)
if((X1train == X3train).all()):
    print("x1 and x3")
else:
    print("diff x1 and x3")
if((X2train == X4train).all()):
    print("x2 and x4")
else:
    print("different x2 and x4")
if((ytrain == ytrain_inception).all()):
    print("y train same")
else:
    print("y train different")
    
model = call(len(vocabulary), max_length)
model.fit([X1train, X3train, X2train], ytrain, epochs=1, verbose=2, batch_size=1000)
model.save('model_fusion'+ '.h5')


#The model loss is 5.9271
#451/451 - 318s - loss: 5.5570
#451/451 - 307s - loss: 5.4065 - for attention
