# Monocular_Depth_Estimation

## Monocular Depth Estimation Project for Neural Networks at Notre Dame

For this project, I am researching current methods of monocular depth estimation. In general, Monocular Depth Estimation is the computer vision task that predicts the depth value of an image, given a single RGB image as input. This type of deep learning is very computationally expensive, but it is something that we use in my lab - the autonomous systems lab, since we work with UAVs that may need to use monocular depth estimators for image segmentation or obstacle avoidance. Because my primary research is on the software engineering side of cyber-physical systems, I am exploring how these types of networks are built. 

For the first stage of the project, I was able to build my own monocular depth estimator from scratch in tensorflow, based off of the U-NET Convolutional Network. This network was a state of the art network introduced for image segmentation, and it works quite well when trained in a supervised manner with images with ground truth depth maps. The architecture that I built from scratch follows the general guidelines described in the paper: https://arxiv.org/pdf/1505.04597.pdf . 


