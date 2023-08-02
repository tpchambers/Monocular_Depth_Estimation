# Monocular_Depth_Estimation Final Project
# Author - Theo Chambers 
# Click here for the [final solution](#Final-Solution).

## Monocular Depth Estimation Project for Neural Networks at Notre Dame Introduction

For this project, I am researching current methods of monocular depth estimation. In general, Monocular Depth Estimation is the term for a computer vision framework that predicts the depth of an image given a single RGB image as input. This type of deep learning is very computationally expensive, but it is something that we use in my lab - the autonomous systems lab. We work with UAVs that may need to use monocular depth estimators for image segmentation or obstacle avoidance in a variety of circumstances. Because my primary research is on the software engineering side of cyber-physical systems, I am exploring how these types of networks are built for the first time.

For the first stage of the project, I was able to build my own monocular depth estimator from scratch in tensorflow, based off of the U-NET Convolutional Network. After building my U-NET model, I tested the architecture on the oxford pets keras dataset, with the KERAS tutorial to test its functionality on a more simple dataset. For the second part of this project, I found a better version of the U-NET model that uses an architecture resembling my own model structure. I then re-trained the model on one of the larger datasets I previously mentioned, the NYU data set: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html . The NYU dataset is a massive dataset comprised of indoor images along with a ground truth depth label per each image. This set has been widely used in computer vision research and is a benchmark for most monocular depth estimation papers that I have seen. For my first project solution, I wanted to fully understand how to build such models, and be able to train one of the pre-made model examples on a larger and more difficult dataset, which I was able to do. I originally cited the DEMON paper in my earlier dataset submission, and the NYUV2 dataset is a critical dataset mentioned as part of the paper. I then discuss what the next steps for my final project will entail. 


### Scratch U-Net 

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

### Instructions

VALIDATION CODE TO RUN: https://colab.research.google.com/drive/15OGtLp5-RHcZZ9I6zd9_zjrpdV5Xm8xv?usp=sharing
Link showing how I trained the model: https://colab.research.google.com/drive/1z0HSGh-iCFntSEwYC9fzxdgBxVaKrztz?usp=sharing 

Please run the code in the google colab above, which I ran multiple times using a GPU over the past two weeks. In order to access the appropriate file paths, all you have to do is go to this location in your google drive: https://drive.google.com/drive/folders/1S0x6ON5WSBlfyk8j7YtHKxnY9rXswUgH?usp=share_link . This contains the larger dataset that I will be running on this model. Once you click on this, then you need to right click and add a shortcut to your drive like so:

![Screen Shot 2023-04-12 at 2 45 02 AM](https://user-images.githubusercontent.com/69804201/231373584-4198ad3f-7d9f-450d-a0b7-3d8b0b5fb20f.png)

I have tested with a friend, and this should allow you to access the correct path in your own google drive account, when running the colab file! 


#### Introduction:
After building my U-NET model from scratch and fully understanding how a basic depth estimator functions, I found a pre-existing comprehensive model at this link: https://keras.io/examples/vision/depth_estimation/ . I had to adjust the data preprocessing and was able to run this model (which is similar to my U-NET) on the more research oriented NYUV2 depth dataset. It took a significant amount of debugging and playing around with the data to get it to work on the different dataset. The NYUV2 data set that I downloaded is around 4 GB in total, with RGB indoor photos next to a labeled ground truth depth map. 

#### Justification:
The architecture of this model is similar to my original U-NET. However, here, the authors use batch normalization after using leaky relu in the upsampling  and downsampling portions. In class, we learned that batch normalization is able to stabilize the outputs of the feature space, and therefore aid the learning process by normalizing the outputs. This extra step will allow the model to train faster. Additionally, the authors of this example use leaky relu. Leaky relu is able to prevent vanishing gradients by simply returning the input if the input is positive, but if the input is negative, it will return a lesser value, but one that is not equal to 0. Therefore, for negative inputs we will not have the problem of vanishing gradients when the derivative is taken during back propagation. The combination of these two aids the training process, but other than that, the U-NET example 

# Final Solution


## Source Code

If not already done, please follow the instructions explained in [Part 2](#Instructions) to setup the directory properly.

Training portion: https://colab.research.google.com/drive/1z0HSGh-iCFntSEwYC9fzxdgBxVaKrztz#scrollTo=Ht8rnqAWPkmj 

Testing portion:  https://colab.research.google.com/drive/15OGtLp5-RHcZZ9I6zd9_zjrpdV5Xm8xv

## Report

For my final solution, I used the same NYUV2 depth dataset, which as described before is a massive 4 GB dataset comprised of indoor RGB camera photos with ground truth depth maps. For the final solution, I fine-tuned my model on a much larger portion of the dataset, and was able to properly train it on over 25,000 images over the past couple of weeks. This is in comparison to my initial solution where I trained the model on a smaller subset of images. The original solution was also tested using a separate subset of the training samples as the validation, whereas in the final solution I tested on both the actual test set of the dataset (which contains around 700 images), and a separate portion of the training in order to compare to the previous training round. This can be seen in my links above. I believe that the differences are substantial enough to test the difference of my final solution since I am exposing the model to completely different types of images using the actual test set as recommended from the official NYUV2 dataset https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html, and I also tested it on a separate training portion which it had never seen before. 

To test the accuracy I used the same classification as before, or the MSE, which is outputted in the google colab file on the validation set. In researching this topic, most general evaluations are done using MSE, since it can provide a fairly stable comparison between two images. The MSE is simply the squared differences between all the pixels of the two images in comparison. The lower the MSE, the stronger and more robust the model. To allow for consistency, below are classifications (errors) using the MSE 400-500 batch images. I had to use a small test subset in both submissions since each of the images and the model prediction would take up a lot of RAM. Below compare the submission 1 solution to the final solution:

Original MSE on training subset (submission 1):

<img width="572" alt="Screen Shot 2023-05-09 at 10 44 55 PM" src="https://github.com/tpchambers/Monocular_Depth_Estimation/assets/69804201/94628ccb-07dc-4241-9f49-e0ff67262af9">. 


Submission 2 MSE metrics:

updated MSE with fine-tuned model on larger training subset: 

<img width="572" alt="Screen Shot 2023-05-09 at 11 25 01 PM" src="https://github.com/tpchambers/Monocular_Depth_Estimation/assets/69804201/581076b1-ce73-44c8-a690-c2260e8f261e">. 


updated MSE with fine-tuned model using test set: .16

<img width="572" alt="Screen Shot 2023-05-10 at 12 02 57 AM" src="https://github.com/tpchambers/Monocular_Depth_Estimation/assets/69804201/764941af-6ffa-403d-be43-6ae9a4f1fcec">



First of all, I am happy with the MSE. The MSE improved substantially from the original test on the training subset compared with the fine-tuned model tested on a training partition not yet seen. Now, looking at the MSE on the actual test set, we see that it was slightly higher than the training MSE, which is expected, although it is still very low. I believe that the model would have improved substantially if I used a pre-trained autencoder. Although the U-NET I used performed well and was fine-tuned on a large amount of images, I noticed that consistently when I trained the model on and off I was not able to do much better at a certain point in my recent training and it would take a very long time. I think that this would have been improved if I fine-tuned my weights using after freezing an encoder like DENSE-NET or another autencoder, and then kept my decoder blocks. I would then be able to fine tune much more efficiently using a very powerful and state of the art encoder. Additionally, I can say that while U-NETs such as the one I used are often times  applied to regression problems such as depth estimation, it is not the ideal architecture for this task. This is due to the performance challenges inherent to U-NET, since they are deep and complicated models that suffer from training issues, and also the fact that the images used to train the model are not high resolution photos. For true monocular depth estimation at a high level, we would need very high resolution photos which the dataset did not provide.













