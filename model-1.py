from preprocess import get_data

def main():
    #This returns a dictionary. The image_captions is a dictionary {image_name, captions}
    #The features is a dictionary {image_name, pixels}
    #The image shape is 32,32,3
    directory = r'C:/Users/aakan/Desktop/CSCI 1470/Final Project- Image Captioning/dataset/Images'
    file_path = r'C:/Users/aakan/Desktop/CSCI 1470/Final Project- Image Captioning/dataset/captions.txt'
    image_captions, features =  get_data(directory, file_path)
    print(len(features))
    print(len(image_captions))
    

if __name__ == '__main__':
    main()

