#!/usr/bin/env python
#-------------------------------------------------------------------------------
# cnn_grocery_test.py
# logfile naming convention:
# cnn: convolutional neural network
# 32pix: resized image_size
# 10k: number of images used (training and testing together)
# 2convpass: number of times the image is run through the convolution and pooling block
# > change the name of the log file to refect the chosen settings.... <
#-------------------------------------------------------------------------------
from cnn_grocery_util2 import *
logfile = '_CNN_32pix_10k_2convpass.txt'

#-------------------------------------------------------------------------------
#settings
n_epoch = 50                    #50, 100
image_size = 32                 #32, 64
conv_pass = 2                   #1, 2
n_classes = 5 
#-------------------------------------------------------------------------------


labels,filenames = load_data()
#create_pickle(labels, filenames, n_epoch, n_classes, image_size, conv_pass, logfile)
test(labels, filenames, n_epoch, n_classes, image_size, conv_pass, logfile)

#clean up
for afile in glob.glob("grocery-classifier.*"):
    os.remove(afile)
for afile in glob.glob("grocery_dataset*.*"):
    os.remove(afile)
