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
import pandas as pd
import time

IPimgfolder = '/data/shared/suhit/pgimages'
ganOPimgfolder = '/data/shared/suhit/pgfakes'
#IPrandomfolder1 = 'C:\\DragonBallZ\\Summer2018\\Video_Image_Processing_GANS\\sharedVBOX\\nf'
#OPrandomfolder2 = 'C:\\DragonBallZ\\Summer2018\\Video_Image_Processing_GANS\\sharedVBOX\\nffakes'

targetGraphNameWithExt = '/data/shared/suhit/ssi_max_plot.png'

#full path where the csv is to be kept. This csv contains for every input image , the image which it is most similar to. 
loc = '/data/shared/suhit/output_ssimax.csv'


from os import listdir
from os.path import isfile, join
ipfiles = [f for f in listdir(IPimgfolder) if isfile(join(IPimgfolder, f))]

def findssi (imageA , imageB):
    s = ssim(imageA,imageB)
    return s

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
max_ssi_val_list = []
max_ssi_ip_op_img_name_dict = {}

print("start iterating over input images ----------------")
#iterating over the input images 

t_ini = time.time()
for ip in l1:
    ippath = os.path.join(IPimgfolder,ip)
    img1 = cv2.imread(ippath,0)
    img1.resize(64,64)
    # to keep the values of the file names and the ssi values ; both would have the same values 
    sortedfilenamesList = []
    ssivalList = []
    file_list = os.listdir(ganOPimgfolder)
    #iterating over the output images 
    print("Input file: '",ip,"' in progress")
    for file in l2:
        ganoppath = os.path.join(ganOPimgfolder,file)
        img2 = cv2.imread(ganoppath,0)       
        sortedfilenamesList.append(file)
        s = findssi(img1,img2)
        ssivalList.append(s)
    # picks the max value of ssi and gets the index. Stores it in the "max...." list created earlier
    ssi_max_value = max(ssivalList)
    max_index = ssivalList.index(ssi_max_value)
    max_ssi_val_list.append(ssi_max_value)
    max_ssi_ip_op_img_name_dict[ip] = sortedfilenamesList[max_index]
print("end iterating over input images -----------------")
t_fin = time.time()
t_diff = t_fin-t_ini
print("Total time taken = ",t_diff)

plt.title('max similarity plot')
plt.plot(max_ssi_val_list,'.-')
plt.savefig(targetGraphNameWithExt,format = 'png')
print("Plot created successfully at ",targetGraphNameWithExt)

(pd.DataFrame.from_dict(data=max_ssi_ip_op_img_name_dict, orient='index')
   .to_csv(loc, header=False))
print("csv saved successfully at ",loc)
print("--------------------SSI MAX PLOT----------------------------------")