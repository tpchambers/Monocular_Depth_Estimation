# Monocular_Depth_Estimation

## Monocular Depth Estimation Project for Neural Networks at Notre Dame

For this project, I am researching current methods of monocular depth estimation. In general, Monocular Depth Estimation is the computer vision task that predicts the depth value of an image, given a single RGB image as input. This type of deep learning is very computationally expensive, but it is something that we use in my lab - the autonomous systems lab, since we work with UAVs that may need to use monocular depth estimators for image segmentation or obstacle avoidance. Because my primary research is on the software engineering side of cyber-physical systems, I am exploring how these types of networks are built. 

For the first stage of the project, I was able to build my own monocular depth estimator from scratch in tensorflow, based off of the U-NET Convolutional Network. This network was a state of the art network introduced for image segmentation, and it works quite well when trained in a supervised manner with images with ground truth depth maps. The architecture that I built from scratch follows the general guidelines described in the paper: https://arxiv.org/pdf/1505.04597.pdf . 

<img width="825" alt="Screen Shot 2023-04-11 at 8 18 26 PM" src="https://user-images.githubusercontent.com/69804201/231315707-9f21185c-b6cd-46f2-bfda-ee8d3527abcb.png">

The network architecture has two different paths as stated by the article: a contrasting path, a bottleneck connecting the two, and an expansive one. The idea is that one should reduce the feature space of the input image, or encode the image to a lower feature space while maintaining depth. Therefore, this follows the idea of an encoder. The contracting path just discussed is then connected to the expansive path through the middle bottleneck, which should retain the prominent features of the layer, similar to a generative model where we seek to reduce the feature space to a different space "z", then rebuild. Thus, the expansive path will restructure the image and learn the true depth of the image, learning the feature representation of the pixels.

For the first part of the project, I replicated the architecture from the paper using tensorflow. I built my own replication of the simple U-NET structure from scratch, then incorporated my architecture into a KERAS tutorial on the oxford pets dataset. My network architecture was built as follows: 


One can see that the simple architecture follows the exact specifications of the paper. The contracted path has two 3x3 convolutions followed by a ReLU activation, and a 2x2 max pooling with a stride of 2 for the downsampling step, doubling the features as we progress. The expansive path upsamples the features (as described by the paper), where we have a 2x2 convolution block, reducing the features by two, followed by a concatenation, then the two convolutional blocks. In KERAS, I found that CONV2dtranspose upsamples by creating features, and this can actually be larned.

