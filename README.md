# Monocular_Depth_Estimation

## Monocular Depth Estimation Project for Neural Networks at Notre Dame

For this project, I am researching current methods of monocular depth estimation. In general, Monocular Depth Estimation is the computer vision task that predicts the depth value of an image, given a single RGB image as input. This type of deep learning is very computationally expensive, but it is something that we use in my lab - the autonomous systems lab, since we work with UAVs that may need to use monocular depth estimators for image segmentation or obstacle avoidance. Because my primary research is on the software engineering side of cyber-physical systems, I am exploring how these types of networks are built for the first time. 

For the first stage of the project, I was able to build my own monocular depth estimator from scratch in tensorflow, based off of the U-NET Convolutional Network. I then built my U-NET model and tested it on the oxford pets keras dataset, with the KERAS tutorial. For the second part of this project, I found a better version of the U-NET model that uses a similar version of my own. I re-trained the model on one of the larger datasets I previously mentioned, the NYU data set: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html . The NYU dataset is a massive dataset comprised of indoor images along with a ground truth depth label. This set has been widely used in computer vision research and is a benchmark for most depth estimation papers. For my first project solution, I wanted to fully understand how to build such models, and be able to train one of the pre-made model examples on a larger and more difficult dataset, which I was able to do. 


### Part 1 

CODE TO RUN : https://colab.research.google.com/drive/19m7JwNRthMrGmxOjdrStyZTWtb15U82G?usp=sharing

Beginning with my scratch implementation and training of a simple U-NET model, this network was originally a state of the art network introduced for image segmentation, and it works quite well when trained in a supervised manner with images with ground truth depth maps. The architecture that I built from scratch follows the general guidelines described in the paper: https://arxiv.org/pdf/1505.04597.pdf . 

<img width="825" alt="Screen Shot 2023-04-11 at 8 18 26 PM" src="https://user-images.githubusercontent.com/69804201/231315707-9f21185c-b6cd-46f2-bfda-ee8d3527abcb.png">

The network architecture has two different paths as stated by the article: a contrasting path, a bottleneck connecting the two, and an expansive one. The idea is that one should reduce the feature space of the input image, or encode the image to a lower feature space while maintaining depth. This follows the structure of an encoder. The contracting path (just discussed) is then connected to the expansive path through the middle bottleneck, which should retain the prominent features of the layer, similar to a generative model where we seek to reduce the feature space to a different space, then reconstruct. Thus, the expansive path will restructure the image and learn the true depth of the image, learning the feature representation of the pixels.

#### Justication:
For the first part of the project, I replicated the architecture from the paper using tensorflow. I built my own replication of the simple U-NET structure using only standard tensorflow functions, then incorporated my architecture into a KERAS tutorial on the oxford pets dataset. My network architecture was built as follows: 

One can see that the simple architecture follows the exact specifications of the paper. The contracted path has two 3x3 convolutions followed by a ReLU activation, and a 2x2 max pooling with a stride of 2 for the downsampling step, doubling the features as we progress. The expansive path upsamples the features (as described by the paper), where we have then have a 2x2 convolution block, reducing the features by two, followed by a concatenation, then two more convolutional blocks. It took me some time to get a close replication of the exact model from scratch. Here is the architecture that I designed below! 

![Screen Shot 2023-04-11 at 9 06 16 PM](https://user-images.githubusercontent.com/69804201/231320889-4e8ad597-7cbf-4543-9387-3db674a3d92c.png)

For my initial experiment we use the adam optimizer, which tends to perform very well for a variety of problems. We also use the loss function of "sparse_categorical_crossentropy", which is utilized when we have truth labels properly formated. This is used in multi-class classifications problems that have single-labels, such as this dataset. I followed the example from KERAS in implementing this code, and I have attached the sources to the google colab. 

I ran the code for 100 Epochs which took a decent amount of time due to the structure of the U-NET. For the purposes of the preliminary phase of the final project, I trained in google colab, although for my final submission I plan on switching to the CRC, since even a simple U-NET took a very long time. Below are the results from the training over a smaller 100 Epochs:

![Screen Shot 2023-04-12 at 2 17 18 AM](https://user-images.githubusercontent.com/69804201/231367932-45c3751d-f706-40ce-82dd-fcfea15f8d34.png)

We can see that in training the U-NET architecture, the model does not overfit after 100 epochs, and it actually keeps improving. For this model, we use the loss functions provided by the keras tutorial, so really this portion of the project was for me to learn from a detailed level how to implement a monocular depth estimator. 

Here is an example of the visualized prediction, and we see that the resulting prediction is quite nice, although there is plenty of work to be done.
![Screen Shot 2023-04-12 at 2 18 25 AM](https://user-images.githubusercontent.com/69804201/231368128-e2318f91-c84d-4d7e-9efa-d5e8278199d9.png)

### Part 2 

VALIDATION CODE TO RUN: https://colab.research.google.com/drive/15OGtLp5-RHcZZ9I6zd9_zjrpdV5Xm8xv?usp=sharing

After building my U-NET model from scratch and fully understanding how a basic depth estimator functions, I found a pre-existing comprehensive model at this link: https://keras.io/examples/vision/depth_estimation/ . I had to completely change the data preprocessing and was able to run this model (which is similar to my U-NET) on the novel NYU depth dataset. 

#### Justification:
The architecture of this model is similar to my original U-NET. However, here, the authors use batch normalization after using leaky relu in the upsampling  and downsampling portions. In class, we learned that batch normalization is able to stabilize the outputs of the feature space, and therefore aid the learning process by normalizing the outputs. This extra step is a smart addition to the U-NET code. Additionally, the authors of this example use leaky relu. Leaky relu is able to prevent vanishing gradients by simply returning the input if the input is positive, but if the input is negative, it will return a lesser value, but one that is not 0. Therefore, for negative inputs we will not have the problem of vanishing gradients when the derivative is taken during back propagation. The combination of these two aids the training process, but other than that, the U-NET example used for this new dataset resembles my original implementation. 


