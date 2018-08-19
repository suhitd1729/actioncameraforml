#!/usr/bin/python
import os,sys, glob
# rename files based on creation time stamp
# create video based on these images ... cd into the dir where the images are:
# avconv -framerate 25 -f image2 -i image%00d.png -c:v h264 -crf 1 out.mp4


#_src = "/home/realtech/Desktop/ongoing/gan/dcgan/fakes2temp/"
#_src = "/home/suhit/proj/fakes/"
_src = "/data/shared/suhit/pg_gan/"

print("Source folder is " ,_src) 
_ext = ".png"
name = "image"
files = list(filter(os.path.isfile, glob.glob(_src + "*")))
files.sort(key=lambda x: os.path.getmtime(x))

offset = 0                  # 0 to start on epoch 0, n after n images

for i in range (0, len(files)):
   print(i,files[i])
   newname = _src + name + str(i+offset) + _ext
   os.rename(files[i], newname)