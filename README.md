# allResults
This repo contains shows all results generated in academics and research work during my Masters.


# Thesis
### Image inpainting by modified U-Net
Image inpainting for satelite images of Lunar and Martian surface was performed to fill/predict the mising pixel values. Original Deep image Prior method was modified to obtain good results. Results are shown below
![inpaint](https://github.com/VishalPrasadIITGn/allResults/blob/main/inpaint.PNG)
![modified UNet](https://github.com/VishalPrasadIITGn/allResults/blob/main/UNet.PNG)

### Data Augmentation using affine transformation
Data augmentation was performed to generate additional/synthetic training samples to train the model. An example of data augmentation is shown.
![aug](https://github.com/VishalPrasadIITGn/allResults/blob/main/augmenttation.PNG)

### Semi-supervised classification
Two-class image classification with limited no. of samples (less than 100 sample) was performed using augmentation and k-fold cross validation.

### Unsupervised Clustering
Images were cluserted using autoencoders. Both full connected and Convolutional autoencoders were used to perform dimensionality reduction and then clustering techniques were applied.
### Crater detection and Segmentation
Training CNN models to perform crater detection and segmentation using a novel method and architecture to beat the SOTA results. Further details cannot be stated as paper is under preparation.

# Machine Learning
## Obect detection and Counting
Faster RCNN model was trained on custom dataset to perform object detection and counting. The model was trained to detect Cars, Trucks and Humans. This was done as part of the project for [ML course in IIT Gandhinagar](https://nipunbatra.github.io/ml2019/). The results obtained for some test images are shown below.
![results1](https://github.com/VishalPrasadIITGn/objectDetectionAndCounting/blob/main/Screenshot%20(193).png)
![results2](https://github.com/VishalPrasadIITGn/objectDetectionAndCounting/blob/main/Screenshot%20(190).png)
![results3](https://github.com/VishalPrasadIITGn/objectDetectionAndCounting/blob/main/Screenshot%20(187).png)



# Computer-Vision-Algorithms-from-scratch
Implementation of Computer vision algorithms from scratch in MATLAB. The results obtained are also shown.
Following Algorithms are implemented from scratch in MATLAB and their results are also shown.
### 1. Image and video denoising by sparse 3D transform-domain collaborative filtering [Link to Paper](http://www.cs.tut.fi/~foi/GCF-BM3D/)
Image de-noising is performed using 3D transform-domain collaborative filtering.
#### Algorithm
BM3D algorithm is shown below
![BM3D algorithm](https://github.com/VishalPrasadIITGn/Computer-Vision-Algorithms-from-scratch/blob/master/Image%20and%20video%20denoising%20by%20sparse%203D%20transform-domain%20collaborative%20filtering/block%20diagram.PNG)
#### Results
Result for denoising are shown below. 
Image on the left is noisy input image. Image on the right is de-noised image obtained by code.

![Image denoising2](https://github.com/VishalPrasadIITGn/Computer-Vision-Algorithms-from-scratch/blob/master/Image%20and%20video%20denoising%20by%20sparse%203D%20transform-domain%20collaborative%20filtering/results3.PNG)
![Image de-noising](https://github.com/VishalPrasadIITGn/Computer-Vision-Algorithms-from-scratch/blob/master/Image%20and%20video%20denoising%20by%20sparse%203D%20transform-domain%20collaborative%20filtering/results2.PNG)

Here de-noising is performed for higher noise values in input image. In the second image, the amount of noise present is very high, still the code manages to produce decent output and recover patterns present in original images.
![Image denoising 3](https://github.com/VishalPrasadIITGn/Computer-Vision-Algorithms-from-scratch/blob/master/Image%20and%20video%20denoising%20by%20sparse%203D%20transform-domain%20collaborative%20filtering/results4.PNG)

This image shows input image, noisy image, output of first stage and final output of the code.
![Image denoising](https://github.com/VishalPrasadIITGn/Computer-Vision-Algorithms-from-scratch/blob/master/Image%20and%20video%20denoising%20by%20sparse%203D%20transform-domain%20collaborative%20filtering/results1.PNG)


### 2. Panorama creation and Image stitching [Link to paper](https://idp.springer.com/authorize/casa?redirect_uri=https://link.springer.com/content/pdf/10.1007/s11263-006-0002-3.pdf&casa_token=-HqEB0GCx84AAAAA:V9rNRgD7vQ9gh1ufw_-n1aJkc0iirA45qTE8MquuFj73oMKenCjIz2Y4qFeUEpHmr-BDFxWY_0H8S4pRDw)
Combining multiple images and creating a panaroma using extracted features and matching keypoints in both images.
### Input Images
![Input Image 1](https://github.com/VishalPrasadIITGn/Computer-Vision-Algorithms-from-scratch/blob/master/Panaroma%20creation%20and%20Image%20Stitching/im_0.jpg)
![Input Image 2](https://github.com/VishalPrasadIITGn/Computer-Vision-Algorithms-from-scratch/blob/master/Panaroma%20creation%20and%20Image%20Stitching/im_1.jpg)
![Input Image 3](https://github.com/VishalPrasadIITGn/Computer-Vision-Algorithms-from-scratch/blob/master/Panaroma%20creation%20and%20Image%20Stitching/im_2.jpg)
### Results: Output Panorama
![Output Image](https://github.com/VishalPrasadIITGn/Computer-Vision-Algorithms-from-scratch/blob/master/Panaroma%20creation%20and%20Image%20Stitching/resGithub.jpg)
This was created withouth using interpolation to fill in the missing pixel values.

### 3. Edge Detection and Convolution with Gaussian Filter
Convolution with Gaussian Filters and then using Difference of Gaussian filter to perform edge detection.
#### Results
##### Results for convolution with Gaussian Filter with varying sigma
![Convolution with Gussian Filter](https://github.com/VishalPrasadIITGn/Computer-Vision-Algorithms-from-scratch/blob/master/Edge%20detection%20and%20Convolution%20with%20Gaussian%20filter/Convolution%20with%20Gaussian%20results.PNG)

##### Results for Edge Detection
![Edge Detection](https://github.com/VishalPrasadIITGn/Computer-Vision-Algorithms-from-scratch/blob/master/Edge%20detection%20and%20Convolution%20with%20Gaussian%20filter/Edge%20Detections%20using%20Difference%20of%20Gaissian%20results.PNG)

### 4. SIFT (Scale Invariant Feature Transform) for object Recognition.[Link to paper](https://ieeexplore.ieee.org/abstract/document/790410/)
SIFT is a highly cited paper for feature extraction and object recognition. It is implemented from scratch and SIFT features are extracted.
#### Results for SIFT
![SIFT descriptors](https://github.com/VishalPrasadIITGn/Computer-Vision-Algorithms-from-scratch/blob/master/SIFT%20(Scale%20Invariant%20Feature%20Transform)%20for%20Object%20Recognition/SIFT%20results.PNG)

### 5. Stereo image correspondences using Fundamental matrix
Pixel realignement was performed between a set of stereo images.
#### Results
##### Input image 1
![Input image 1](https://github.com/VishalPrasadIITGn/Computer-Vision-Algorithms-from-scratch/blob/master/Stereo%20image%20correspondences%20using%20Fundamental%20matrix/3_1.jpg)
##### Input image 2
![Input image 2](https://github.com/VishalPrasadIITGn/Computer-Vision-Algorithms-from-scratch/blob/master/Stereo%20image%20correspondences%20using%20Fundamental%20matrix/3_2.jpg)
#### Output image (original images were resized to reduce the computation)
Pixels of second image were realigned to resemble/recreate the first image.
![results](https://github.com/VishalPrasadIITGn/Computer-Vision-Algorithms-from-scratch/blob/master/Stereo%20image%20correspondences%20using%20Fundamental%20matrix/results1.PNG)

# Nature_Inspired_Computing-NIC-
Evolutionary Computing algorithms like Genetic Algorithms (GA), Particle Swarm Optimization (PSO), Non-Domination Sorting Algorithms (NSGA-2), Fuzzy Algorithms and ANFIS along with their results. Results are shown in .GIF/.PNG format for all codes.
### Function approximation using Particle Swarm Optimization (PSO) 
#### Results
![PSO](https://github.com/VishalPrasadIITGn/Nature_Inspired_Computing-NIC-/blob/master/Particle_Sworm_Optimization_results.gif)

### NSGA 2 (Non-Domination Sorting Genetic Algorithm 2)
#### Results 
![NSGA 2](https://github.com/VishalPrasadIITGn/Nature_Inspired_Computing-NIC-/blob/master/NSGA_2%20(NIC_A2)%20results.png)


### Portfolio optimization using MOPSO (Multi Objective Particle Swarm Optimization)
#### Results to minimise the risk and maximise the profit
![MOPSO](https://github.com/VishalPrasadIITGn/Nature_Inspired_Computing-NIC-/blob/master/MOPSO_results_main.gif)

### Function approximation using Advanced Neural Fuzzy Inference System (ANFIS)
#### Results to approximate a function using ANFIS
![ANFIS](https://github.com/VishalPrasadIITGn/Nature_Inspired_Computing-NIC-/blob/master/ANFIS_results.gif)


