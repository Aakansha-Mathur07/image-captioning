from preprocess import get_features

def main():
    features =  get_features(r'C:/Users/aakan/Desktop/CSCI 1470/Final Project- Image Captioning/dataset/Images')
    print(len(features))
    print(features)
    

if __name__ == '__main__':
    main()

