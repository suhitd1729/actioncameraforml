import matplotlib
matplotlib.use('Agg')
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image
from skimage import color
from os import listdir
from os.path import isfile, join
import math
import random


targetGraphNameWithExt = '/data/shared/suhit/playground_1nn_grid_plot.png'
imagetitle = "1NN for Playground data" 

ipfolder = "/data/shared/suhit/pgimages"
opfolder = "/data/shared/suhit/pg_gan"


inpImgList= []

def numeric_chars2(x):
    y = x[5:-4]
    return(int(y))

ipfiles = [f for f in os.listdir(ipfolder)]
opfiles = [f for f in os.listdir(opfolder)]

l1 = list(np.random.choice(ipfiles,64,replace=False))
l2 = list(np.random.choice(opfiles,64,replace=False))
l2 = sorted(l2,key = numeric_chars2)

def populateImageList(inpImgList,out_img,size):
    listSize = [(size*i)/8 for i in range(0,8)]
    imgList = []
    outImgList = [] 
    for col in listSize:
        for row in listSize:
            area = (row,col,row+(size/8),col+(size/8))
            out_img_grid = out_img.crop(area)
            outImgList.append(out_img_grid)
    imgList.extend(inpImgList)
    imgList.extend(outImgList)
    return imgList

def standardizeImageList(imgList):
    new_width,new_height = 64,64
    imgList = [i.resize((new_width, new_height), Image.ANTIALIAS) for i in imgList]
    return imgList
    
def imgToArray(imgList):
    npArray = []
    for i in imgList:
        I = np.asarray(i.convert('L'))
        npArray.append(I)
    return npArray
        
def compute2Norm(u,v):
    i = u - v
    return np.linalg.norm(i) 


def createDictOf2NormValues(npArray):
    startIndex = 0
    d = {}
    for i in npArray:
        lst = []
        for j in npArray:
            val = compute2Norm(i,j)
            if val == 0:
                val = float("inf")
            lst.append(val)
        d[startIndex] = lst
        startIndex+=1
    return d

def getAccuracy(d):
    count = 0 
    for k in d.keys():
        valList = d[k]
        minIndex = valList.index(min(valList))
        if (k <= 63 and minIndex <=63) or (k > 63 and minIndex > 63):
            count+=1
    #print("count ",count)
    acc = (count/128.0)*100 
    return acc        
        

inpImgList = []
for ip in l1:
    ippath = os.path.join(ipfolder,ip)
    img = Image.open(ippath)
    inpImgList.append(img)

accList = []
d = {}
for opimage in l2:
    oppath = os.path.join(opfolder,opimage)
    out_img = Image.open(oppath)
    size = out_img.size[0]
    imgList = populateImageList(inpImgList,out_img,size)
    imgList = standardizeImageList(imgList)
    npArray = imgToArray(imgList)
    d = createDictOf2NormValues(npArray)
    acc = getAccuracy(d)
    accList.append(acc)


plt.title(imagetitle)
plt.xlabel('output image index')
plt.ylabel('2 Norm value')
plt.plot(accList,'.-')
plt.savefig(targetGraphNameWithExt,format = 'png')
print("Plot created successfully at ",targetGraphNameWithExt)