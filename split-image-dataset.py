#The dataset has been split using the following code. Do not run again. Simply donwload the dataset.
import splitfolders

directory = r'C:/Users/aakan/Desktop/CSCI 1470/Final Project- Image Captioning/dataset/Images'
splitfolders.ratio(directory, output="dataset", seed=1337, ratio=(.8, .2), group_prefix=None)