from preprocess import get_data

def main():
    #This returns a dictionary. The image_captions is a dictionary {image_name, captions}
    #The features is a dictionary {image_name, pixels}
    #The image shape is 32,32,3
    
    '''directory = r'/Users/hilonimehta/Desktop/image-captioning-main/dataset/Images'
    #file_path = r'C:/Users/aakan/Desktop/CSCI 1470/Final Project- Image Captioning/dataset/captions.txt'
    file_path = r'/Users/hilonimehta/Desktop/image-captioning-main/dataset/captions.txt' 
    image_captions, features =  get_data(directory, file_path)
    print(len(features))
    print(len(image_captions)) '''
    
     #This returns a dictionary. The image_captions is a dictionary {image_name, captions}
    #The features is a dictionary {image_name, pixels}
    #The image shape is 224,224,3
    train_directory = r'dataset/train/Images'
    train_file_path = r'dataset/train_captions.txt'
    train_captions, train_features =  get_data(train_directory, train_file_path,model_type=None)
    print(len(train_features))
    print(len(train_captions))
    
    test_directory = r'dataset/test/Images'
    test_file_path = r'dataset/test_captions.txt'
    test_captions, test_features =  get_data(test_directory, test_file_path,model_type=None)
    print(len(test_features))
    print(len(test_captions))
    

if __name__ == '__main__': 
    main()  
 
