from preprocess import get_datafrom keras.applications.vgg16 import VGG16from keras.applications.vgg16 import preprocess_inputfrom keras.applications.vgg16 import decode_predictionsfrom keras.layers import Inputimport keras.models def vgg16(features):    #Background - vgg16 has 16 convolutional layers. It has uniform architecture.Parameters-138 million parameters#Input- size:224-3*3 conv, 64,3*3 conv, 64, pool, size:112-3*3 conv, 128,3*3 conv, 128, pool, size 56- 3layers, size-28, 3 layers, size 14, 3 layers, size 7, 3 dense layers #loading vgg16 model    print(len(features)) #include_top=False,    #model = VGG16(input_tensor=Input((224,224,3)), pooling="avg")#include_top=False remove the last 2 dense layers    #try with and without pooling... or pooling="avg"/"max" and play around with it    model = VGG16(input_tensor=Input((224,224,3)), pooling="avg")    model_last_remove = keras.models.Model(inputs=model.inputs, outputs=model.layers[-2].output)#removuing the last dense layer    vgg_pred = {}    count = 0    for image_array in features:        count += 1        image = features[image_array]         image = preprocess_input(image)                 prob = model_last_remove.predict(image, verbose=0)         print(len(prob))        print(len(prob[0]))              print(image_array)               vgg_pred[image_array] = prob         #if(count == 1):        #    break            #uncomment to see predictions i.e. decode predictions9Also change model_last-remove to remove in that case)        #labels = decode_predictions(prob)            return vgg_pred   def main():    #This returns a dictionary. The image_captions is a dictionary {image_name, captions}    #The features is a dictionary {image_name, pixels}    #The image shape is 224,224,3        train_directory = r'dataset/train/Images'    train_file_path = r'dataset/train_captions.txt'    model_type = 'vgg'    train_captions, train_features =  get_data(train_directory, train_file_path,model_type)    print(len(train_features))    print(len(train_captions))    #image_captions, features =  get_data(directory, file_path,model_type)    #print(len(features))    #print(len(image_captions))     #print(image_captions)    #print(features)    test_directory = r'dataset/test/Images'    test_file_path = r'dataset/test_captions.txt'    test_captions, test_features =  get_data(test_directory, test_file_path, model_type=None)      print(len(test_features))    print(len(test_captions))    #inception_pred = inception(train_features)        '''directory = r'/Users/hilonimehta/Desktop/image-captioning-main/dataset/Images'    file_path = r'/Users/hilonimehta/Desktop/image-captioning-main/dataset/captions.txt'    model_type = 'vgg'    image_captions, features =  get_data(directory, file_path,model_type)'''        vgg_pred = vgg16(train_features)    return vgg_pred        if __name__ == '__main__':    main()