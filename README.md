#Smart Image Sampling 
This project includes developing smart sampling techniques to sample key images (frames) from long video sequences. The objective was to ensure that the images so obtained have minimum redundancy among them and that the auxiliary information about the video capture event, such as location, time, annotations by the video recorder, camera focus, etc., is integrated with the image content.
The processed images are then fed into a Generative Adversarial Network (GAN) model to generate synthetic images that are similar to the input images.

In order to compare the input and the output images , the following techniques have been used: 
1)  Kernel Maximum Mean Discrepancy (KMMD)
2) Structural Similarity Index 
3) 1 Nearest Neighbour

Reference : https://arxiv.org/pdf/1802.03446.pdf
 
Following steps are undertaken to run the experiments : 
A) Conversion of video to a sequence of png images  

avconv -i /data/shared/videos/demo.mp4 -r framerate -f image2 /home/suhitdat/input_images/%04d.png

B) Feeding the input images to a GAN. 
This will spit the output in a directory specified inside the python code.
 
python3 main_v2_modified.py --dataroot "directory containing input images folder"  --dataset "input images folder name" --niter "num of epochs to train for" --cuda

eg:  main_v2_modified.py --dataroot /data/shared/images/grocerystore/ --dataset beverages --cuda --niter 25

C) Renaming the image outputs as per timestamp
This renames the file names from --epoch-- format to plain "image%%" where %% represents the numeric value. Output image folder is specified inside the python code.

python3 rename.py 

D) Computing Sharpness of Output images by GAN 
This will compute the sharpness of the output images. 
Output image folder is specified inside the python code.
Note: Each output image comprises 8*8 images. Batchsize of 64 images specified while running the GAN.
  
grid_sharpness_measure.py
 
E) Running the Techniques/Modules explained above.

All the python codes have an option to specify the input images and the output images folder. 
eg: 
ipfolder = "/data/shared/images/grocerystore/beverages"
opfolder = "/data/shared/suhit/beverages_gan"

The name/location of the output plot can be specified inside the python code. 

To run a structural similarity index comparison between input and output images: 
python3 ssi_grid.py

To run a Kernel Maximum Mean Discrepancy comparison between input and output images: 
python3 kmmd_grid.py

To run a 1 Nearest Neighbor comparison between input and output images: 
python3 1nn_grid.py

For a sample of the outputs obtained by method "*", check the plots_* folder  