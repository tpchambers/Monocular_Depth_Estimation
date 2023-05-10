# Monocular_Depth_Estimation Final Project
# Author - Theo Chambers 
# [Final Solution](#Final-Solution)
## Note - Please run the code through the google colab links below:
## Part 1 Link: https://colab.research.google.com/drive/19m7JwNRthMrGmxOjdrStyZTWtb15U82G?usp=sharing
## Part 2 Link: https://colab.research.google.com/drive/15OGtLp5-RHcZZ9I6zd9_zjrpdV5Xm8xv?usp=sharing
## For the second part of this submission, I saved the model checkpoints as I trained on GPU to the google drive folder shared called "NNProject_TheoChambers" (link for this along with instructions below in [Part 2](#Part-2). 

## Monocular Depth Estimation Project for Neural Networks at Notre Dame

For this project, I am researching current methods of monocular depth estimation. In general, Monocular Depth Estimation is the term for a computer vision framework that predicts the depth of an image given a single RGB image as input. This type of deep learning is very computationally expensive, but it is something that we use in my lab - the autonomous systems lab. We work with UAVs that may need to use monocular depth estimators for image segmentation or obstacle avoidance in a variety of circumstances. Because my primary research is on the software engineering side of cyber-physical systems, I am exploring how these types of networks are built for the first time.

For the first stage of the project, I was able to build my own monocular depth estimator from scratch in tensorflow, based off of the U-NET Convolutional Network. After building my U-NET model, I tested the architecture on the oxford pets keras dataset, with the KERAS tutorial to test its functionality on a more simple dataset. For the second part of this project, I found a better version of the U-NET model that uses an architecture resembling my own model structure. I then re-trained the model on one of the larger datasets I previously mentioned, the NYU data set: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html . The NYU dataset is a massive dataset comprised of indoor images along with a ground truth depth label per each image. This set has been widely used in computer vision research and is a benchmark for most monocular depth estimation papers that I have seen. For my first project solution, I wanted to fully understand how to build such models, and be able to train one of the pre-made model examples on a larger and more difficult dataset, which I was able to do. I originally cited the DEMON paper in my earlier dataset submission, and the NYUV2 dataset is a critical dataset mentioned as part of the paper. I then discuss what the next steps for my final project will entail. 


### Part 1 

CODE TO RUN : https://colab.research.google.com/drive/19m7JwNRthMrGmxOjdrStyZTWtb15U82G?usp=sharing

Beginning with my scratch implementation and training of a simple U-NET model, this network was originally a state of the art network introduced for image segmentation, and it works quite well when trained in a supervised manner with images with ground truth depth maps if the images are relatively simple. The architecture that I built from scratch follows the general guidelines described in the paper: https://arxiv.org/pdf/1505.04597.pdf . I was able to read the paper and implement the model based on its architecture. 

<img width="825" alt="Screen Shot 2023-04-11 at 8 18 26 PM" src="https://user-images.githubusercontent.com/69804201/231315707-9f21185c-b6cd-46f2-bfda-ee8d3527abcb.png">

#### Discussion
The network architecture has two different paths as stated by the article: a contrasting path, a bottleneck connecting the two, then an expansive one. The idea is that one should reduce the feature space of the input image, or encode the image to a lower feature space while maintaining depth. This follows the structure of an encoder as seen in class. The contracting path (just discussed) is then connected to the expansive path through the middle bottleneck, which should retain the prominent features of the layer, similar to a generative model where we seek to reduce the feature space to a different space, then reconstruct. Thus, the expansive path will restructure the image and learn the true depth of the image, learning the feature representation of the pixels in an ideal training session.

#### Justication:
For the first part of the project, I replicated the architecture from the paper using tensorflow. I built my own replication of the simple U-NET structure using only standard tensorflow functions, then incorporated my architecture into a KERAS tutorial on the oxford pets dataset. My network architecture was built as follows: 

One can see that the simple architecture follows the exact specifications of the paper. The contracted path has two 3x3 convolutions followed by a ReLU activation, and a 2x2 max pooling with a stride of 2 for the downsampling step, doubling the features as we progress. The expansive path upsamples the features (as described by the paper), where we have then have a 2x2 convolution block, reducing the features by two, followed by a concatenation, then two more convolutional blocks. It took me some time to get a close replication of the exact model from scratch. Here is the architecture that I designed below! 

![Screen Shot 2023-04-11 at 9 06 16 PM](https://user-images.githubusercontent.com/69804201/231320889-4e8ad597-7cbf-4543-9387-3db674a3d92c.png)

#### Experiment Details
For my initial experiment we use the adam optimizer, which tends to perform very well for a variety of problems. We also use the loss function of "sparse_categorical_crossentropy", which is utilized when we have truth labels (same as categorical cross entropy, but dependent on the data preparation). This is used in multi-class classifications problems that have single-labels, such as this dataset. I followed the example from KERAS in implementing this code, and I have attached the sources to the google colab. 

I ran the code for 100 Epochs which took a decent amount of time due to the structure of the U-NET. For the purposes of the preliminary phase of the final project, I trained in google colab, although for my final submission I plan on switching to the CRC on a different model, since even a simple U-NET took a very long time. Below are the results from the training over a smaller 100 Epochs:

![Screen Shot 2023-04-12 at 2 17 18 AM](https://user-images.githubusercontent.com/69804201/231367932-45c3751d-f706-40ce-82dd-fcfea15f8d34.png)

We can see that in training the U-NET architecture, the model does not overfit after 100 epochs, and it actually keeps improving. For this model, we use the loss functions provided by the keras tutorial, so really this portion of the project was for me to learn from a detailed level how to implement a monocular depth estimator. 

Here is an example of the visualized prediction, and we see that the resulting prediction is quite nice, although there is plenty of work to be done.
![Screen Shot 2023-04-12 at 2 18 25 AM](https://user-images.githubusercontent.com/69804201/231368128-e2318f91-c84d-4d7e-9efa-d5e8278199d9.png)

I did not evaluate this on a test set, since this was an exercise for me to learn how to implement a monocular depth estimator from scratch. In the next part below, I will test a pre-made model on a different and more novel dataset.

### Part 2 

VALIDATION CODE TO RUN: https://colab.research.google.com/drive/15OGtLp5-RHcZZ9I6zd9_zjrpdV5Xm8xv?usp=sharing
Link showing how I trained the model: https://colab.research.google.com/drive/1z0HSGh-iCFntSEwYC9fzxdgBxVaKrztz?usp=sharing 

INSTRUCTIONS:
Please run the code in the google colab above, which I ran multiple times using a GPU over the past two weeks. In order to access the appropriate file paths, all you have to do is go to this location in your google drive: https://drive.google.com/drive/folders/1S0x6ON5WSBlfyk8j7YtHKxnY9rXswUgH?usp=share_link . This contains the larger dataset that I will be running on this model. Once you click on this, then you need to right click and add a shortcut to your drive like so:

![Screen Shot 2023-04-12 at 2 45 02 AM](https://user-images.githubusercontent.com/69804201/231373584-4198ad3f-7d9f-450d-a0b7-3d8b0b5fb20f.png)

I have tested with a friend, and this should allow you to access the correct path in your own google drive account, when running the colab file! 


#### Introduction:
After building my U-NET model from scratch and fully understanding how a basic depth estimator functions, I found a pre-existing comprehensive model at this link: https://keras.io/examples/vision/depth_estimation/ . I had to adjust the data preprocessing and was able to run this model (which is similar to my U-NET) on the more research oriented NYUV2 depth dataset. It took a significant amount of debugging and playing around with the data to get it to work on the different dataset. The NYUV2 data set that I downloaded is around 4 GB in total, with RGB indoor photos next to a labeled ground truth depth map. 

#### Justification:
The architecture of this model is similar to my original U-NET. However, here, the authors use batch normalization after using leaky relu in the upsampling  and downsampling portions. In class, we learned that batch normalization is able to stabilize the outputs of the feature space, and therefore aid the learning process by normalizing the outputs. This extra step will allow the model to train faster. Additionally, the authors of this example use leaky relu. Leaky relu is able to prevent vanishing gradients by simply returning the input if the input is positive, but if the input is negative, it will return a lesser value, but one that is not equal to 0. Therefore, for negative inputs we will not have the problem of vanishing gradients when the derivative is taken during back propagation. The combination of these two aids the training process, but other than that, the U-NET example used for this new dataset closely resembles my implementation. 

#### Classification Accuracy:

To compare the relative accuracy I used MSE, which is outputted in the google colab file on the validation set. In researching this topic, most general evaluations are done using MSE, since it can provide a fairly stable comparison between two images. The MSE is simply the squared differences between all the pixels of the two images in comparison. The lower the MSE, the stronger and more robust the model. I used a smaller portion of the image training set as the validation, since the set is so large and I could not run it in on the CRC in time. For the final project I will change this to have a more comprehensive evaluation using more GPUS.


#### Commentary 
The initial results on the validation set is adequate and should perform well with a more challenging dataset. With the smaller training set, I was able to achieve good performance and set the foundation for the larger dataset.

## Final Solution
Here will be the final solution.

