from extractText import get_data

train_caption_filename = r'dataset/train_captions.txt'
train_image_filename = r'train_features.pkl'
test_caption_filename = r'dataset/test_captions.txt'
test_image_filename = r'test_features.pkl'
train_captions, train_vocabulary, train_image_features = get_data(train_caption_filename, train_image_filename, 'train')
test_captions, test_vocabulary, test_image_features = get_data(test_caption_filename, train_image_filename, 'test')