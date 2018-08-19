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

ipfolder = "/data/shared/suhit/pgimages"
opfolder = "/data/shared/suhit/pg_gan"
targetGraphNameWithExt = '/data/shared/suhit/kmmd_pg.png'
plot_title = 'Kernel Max Mean Discrepancy for Playground'
dct = {}
niter = 100 #specifies the number of iterations that need to be run 

def numeric_chars2(x):
    y = x[5:-4]
    return(int(y))

file_list = os.listdir(opfolder)
l1 = sorted(file_list, key = numeric_chars2)
ipfiles = [f for f in os.listdir(ipfolder)]

def kf(a,b):
    I1 = np.asarray(a.convert('L'))
    I2 = np.asarray(b.convert('L'))
    i = I1 - I2
    #revsigma=1.0/100000000 #10^8
    revsigma=1.0/10000000 #10^7
    val = np.exp(-0.5*revsigma*np.linalg.norm(i)**2)
    return val

def getRandomFromOutput(dct,img):
    A1,A2 = np.random.choice(list(dct.keys()),2,replace=False)
    xg = img.crop(dct[A1])
    xgprime = img.crop(dct[A2])
    return xg,xgprime

def computeKernel(xr,xrprime,xg,xgprime):
    exp1 = kf(xr,xrprime)
    exp2 = -2 * kf(xr,xg)
    exp3 = kf(xg,xgprime)
    exp = (exp1 + exp2 + exp3)**0.5
    if math.isnan(exp):
        exp = 0 
    return exp

def getRandomFromInput():
    v1,v2 = np.random.choice(ipfiles,2,replace=False)
    inpi1 = os.path.join(ipfolder,v1)
    inpi2 = os.path.join(ipfolder,v2)
    xr = Image.open(inpi1)
    xrprime = Image.open(inpi2)
    return xr,xrprime

def populatedct(size):
    listSize = [(size*i)/8 for i in range(0,8)]
    startIndex = 1
    for col in listSize:
        for row in listSize:
            area = (row,col,row+(size/8),col+(size/8))
            dct[startIndex] = area
            startIndex+=1
    print(dct)
    return dct

# Reshapes every incoming image to 64*64 
def reshapeImg(img):
    new_width,new_height = 64,64
    img = img.resize((new_width, new_height), Image.ANTIALIAS)
    return img
    

new_height,new_width = 0,0
kernelList = []
emptydic = True 
for op in l1:
    xr,xrprime = getRandomFromInput()
    oppath = os.path.join(opfolder,op)
    img = Image.open(oppath)
    size = img.size[0]
    if emptydic :
        dct = populatedct(size)
        emptydic = False
    xr = reshapeImg(xr)
    xrprime = reshapeImg(xrprime)
    kernelAvglist = []
    for i in range(niter):
        xg,xgprime = getRandomFromOutput(dct,img)
        xg = reshapeImg(xg)
        xgprime = reshapeImg(xgprime)
        kernelval = computeKernel(xr,xrprime,xg,xgprime)
        kernelAvglist.append(kernelval) 
    kernelList.append(sum(kernelAvglist)/len(kernelAvglist))


plt.title(plot_title)
plt.xlabel('Output Image Index')
plt.ylabel('KMMD Value')
plt.xticks(rotation=45)
plt.plot(kernelList,'.-')
plt.savefig(targetGraphNameWithExt,format = 'png')

print("Plot created successfully at ",targetGraphNameWithExt)