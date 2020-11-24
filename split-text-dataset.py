from os import listdir

def main():
    '''
    with open(r'dataset/captions.txt', 'r') as f:
        captions = f.read()
    for name in listdir(r'dataset/train/Images'):
        for line in captions.split('\n'):
            line_data = line.split(',')
            file_name = line_data[0]
            if(name==file_name):
                train_captions = open(r'dataset/train_captions.txt', 'a')
                train_captions.writelines(line + "\n")
                train_captions.close()
    '''
    with open(r'dataset/captions.txt', 'r') as f:
        captions = f.read()
    for name in listdir(r'dataset/test/Images'):
        for line in captions.split('\n'):
            line_data = line.split(',')
            file_name = line_data[0]
            if(name==file_name):
                test_captions = open(r'dataset/test_captions.txt', 'a')
                test_captions.writelines(line + "\n")
                test_captions.close()

if __name__ == '__main__':
    main()
    