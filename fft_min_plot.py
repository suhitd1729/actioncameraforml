import matplotlib
matplotlib.use('Agg')
from PIL import Image
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from skimage import color
from skimage.measure import structural_similarity as ssim
import random
import time 

IPimgfolder = '/data/shared/suhit/pgimages'
ganOPimgfolder = '/data/shared/suhit/pgfakes'
#IPrandomfolder1 = 'C:\\DragonBallZ\\Summer2018\\Video_Image_Processing_GANS\\sharedVBOX\\nf'
#OPrandomfolder2 = 'C:\\DragonBallZ\\Summer2018\\Video_Image_Processing_GANS\\sharedVBOX\\nffakes'

#full path where the plot is to be kept
targetGraphNameWithExt = '/data/shared/suhit/fft_min_plot.png'

#full path where the csv is to be kept. This csv contains for every input image , the image which it is most similar to. 
loc = '/data/shared/suhit/output_fftmin.csv'

from os import listdir
from os.path import isfile, join
ipfiles = [f for f in listdir(IPimgfolder) if isfile(join(IPimgfolder, f))]

def findfft(imageA , imageB):
    im1 = np.fft.fft2(imageA)
    im1_fshift = np.fft.fftshift(im1)
    mag1 = 20 * np.log(np.abs(im1_fshift))
    
    im2 = np.fft.fft2(imageB)
    im2_fshift = np.fft.fftshift(im2)
    mag2 = 20 * np.log(np.abs(im2_fshift))
    
    sim = np.sqrt(np.sum((mag1-mag2)**2))
    return sim

def numeric_chars1(x):
    y = x[:-4]
    return(int(y))

def numeric_chars2(x):
    y = x[5:-4]
    return(int(y))

file_list1 = os.listdir(IPimgfolder)
l1 = sorted(file_list1, key = numeric_chars1)

file_list2 = os.listdir(ganOPimgfolder)
l2 = sorted(file_list2, key = numeric_chars2)

# to keep the value of the image corresponding to max ssi and correspondingly its image name for the given input image
min_fft_val_list = []
min_fft_ip_op_img_name_dict = {}
print("start iterating over input images ----------------")

#iterating over the input images 
t_ini = time.time()
for ip in l1:
    ippath = os.path.join(IPimgfolder,ip)
    img1 = cv2.imread(ippath,0)
    img1.resize(64,64)
    # to keep the values of the file names and the ssi values ; both would have the same values 
    sortedfilenamesList = []
    fftvalList = []
    file_list = os.listdir(ganOPimgfolder)
    #iterating over the output images 
    print("Input file: '",ip,"' in progress")
    for file in l2:
        ganoppath = os.path.join(ganOPimgfolder,file)
        img2 = cv2.imread(ganoppath,0)       
        sortedfilenamesList.append(file)
        s = findfft(img1,img2)
        fftvalList.append(s)
    # picks the max value of ssi and gets the index. Stores it in the "max...." list created earlier
    fft_min_value = min(fftvalList)
    min_index = fftvalList.index(fft_min_value)
    min_fft_val_list.append(fft_min_value)
    min_fft_ip_op_img_name_dict[ip] = sortedfilenamesList[min_index]
print("end iterating over input images -----------------")
t_fin = time.time()
t_diff = t_fin-t_ini
print("Total time taken = ",t_diff)

plt.title('Similarity Plot (using FFT)')
plt.plot(min_fft_val_list,'.-')
plt.savefig(targetGraphNameWithExt,format = 'png')
print("Plot created successfully at ",targetGraphNameWithExt)

import pandas as pd
(pd.DataFrame.from_dict(data=min_fft_ip_op_img_name_dict, orient='index')
   .to_csv(loc, header=False))
print("csv saved successfully at ",loc)
print("--------------------FFT MIN PLOT----------------------------------")