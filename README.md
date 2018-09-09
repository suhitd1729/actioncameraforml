

# Action cameras - Evaluating image quality and suitability for Machine Learning 

This project aims to evaluate the suitability of images sourced from video (of action cameras) for machine learning applications. The project has three parts:
<br>
**1) Extracting images from video**

We extract individual images at frame rate from the video. Then we select the keyframes from this sequence.

**2) Applying these images to one particular machine learning task: image generation via a Generative Adversarial Network (GAN)**<br>

We train a GAN to reproduce the images from the original video. Our GAN is DCGAN implemented in pytorch, based on the example provided here: https://github.com/soumith/dcgan.torch. See code below for details.

https://github.com/suhitd1729/Action-cameras-evaluating-image-quality-and-suitability-for-machine-learning/blob/master/main_v2_modified.py

**3) Evaluating performance of the output based on established metrics** (Reference : https://arxiv.org/pdf/1802.03446.pdf)<br>

We evaluate the results by comparing the original images from video to the output images of the GAN by the following metrics:<br>
**a)  Kernel Maximum Mean Discrepancy (KMMD)**<br>
**b) Structural Similarity Index**<br> 
**c) 1 Nearest Neighbour**<br>

<br>
*Reference :* https://arxiv.org/pdf/1802.03446.pdf
<br><br>

### Usage 

This project includes developing smart sampling techniques to sample key images (frames) from long video sequences. The objective was to ensure that the images so obtained have minimum redundancy among them and that the auxiliary information about the video capture event, such as location, time, annotations by the video recorder, camera focus, etc., is integrated with the image content.

The processed images are then fed into a *Generative Adversarial Network (GAN) model* to generate synthetic images that are similar to the input images.

### Following steps are undertaken to run the experiments:

**1) Conversion of video to a sequence of png images** <br>

avconv -i /data/shared/videos/demo.mp4 -r framerate -f image2 /home/suhitdat/input_images/%04d.png

**2) Feeding the input images to a GAN** <br>

This will spit the output in a directory specified inside the python code.

python3 **main_v2_modified.py** --*dataroot* "directory containing input images folder"  --*dataset* "input images folder name" --*niter* "num of epochs to train for" --*cuda*
<br>
eg: <br>
python3 main_v2_modified.py --dataroot /data/shared/images/grocerystore/ --dataset beverages --cuda --niter 25

**3) Renaming the image outputs as per timestamp** <br>

This renames the file names from --epoch-- format to plain "image%%" where %% represents the numeric value. Output image folder is specified inside the python code. <br>

python3 rename.py 

**4) Computing Sharpness of Output images by GAN** <br>

This will compute the sharpness of the output images. 
Output image folder is specified inside the python code. <br>
Note: Each output image comprises 8x8 images. Batchsize of 64 images has been specified while running the GAN. <br>  

grid_sharpness_measure.py
 
**5) Running the Techniques/Modules explained above** <br>

All the python codes have an option to specify the input images and the output images folder. 
eg: 
**ipfolder** = "/data/shared/images/grocerystore/beverages"
**opfolder** = "/data/shared/suhit/beverages_gan"

The name/location of the output plot can be specified inside the python code. 

To run a **structural similarity index** comparison between input and output images: 
**python3 ssi_grid.py**

To run a **Kernel Maximum Mean Discrepancy** comparison between input and output images: 
**python3 kmmd_grid.py**

To run a **1 Nearest Neighbor** comparison between input and output images: 
**python3 1nn_grid.py**

For a sample of the outputs obtained by method **"X"**, check the **plots_X** folder  
