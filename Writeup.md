# Dog Breed Classifier with Deep Learning

## Introduction

This project is about preparing a classifier allowing dogs and humans recognition. Given an image of a dog, the algorithm will identify an estimate of the dog's breed. If supplied an image of a human, the code will identify the resembling dog breed.

We will define and train a Convolutional Neural Network  (CNN) using transfer learning, a technique that allows a model developed for a task to be reused as the starting point for another task. CNN network architecture is a specialized setup of interconnected neural network layers used in visal data analysis. CNN model typically consists of convolutional layers, activation function, pooling layers, fully connected layers and normalization layers.

This project is part of XXXXXXXX udacity XXXXXXXX

![Sample Output][images/sample_dog_output.png]


## Objective

The objective of this project is simply trainining an image classifier to recognize different breeds of dogs, then export it for use in a stand alone application.

We will piece together a series of models to perform different tasks; for instance, the algorithm that detects humans in an image will be different from the CNN that infers dog breed. 

The purpose of this article is descibe technical steps and their outcome from start to finish.

## Project Instructions

In order to setup everything to get started, follow [the Project Instructions](README.md).

## Approach

The approach we will take to fulfill our objective can be represented by the following list of steps:

- Step 1: Import Datasets
- Step 2: Detect Humans
- Step 3: Detect Dogs
- Step 4: Create a CNN to Classify Dog Breeds (fromScratch - define a baseline)
- Step 5: Use a CNN to Classify Dog Breeds (using Transfer Learning)
- Step 6: Create a CNN to Classify Dog Breeds (using Transfer Learning)
- Step 7: Write your Algorithm
- Step 8: Test Your Algorithm

## Datasets Import and EDA

The first step is importing dog and human images datasets. We populate a few variables through the use of the load_files function from the scikit-learn library:

- `train_files`, `valid_files`, `test_files` - numpy arrays containing file paths to images
- `train_targets`, `valid_targets`, `test_targets` - numpy arrays containing onehot-encoded classification labels
- `dog_names` - list of string-valued dog breed names for translating labels

There are 8351 total dog images in 133 breed categories. For modeling purposes, the dataset is divided into sub-datasets:
 
- training (6680 dog images).
- validation (835 dog images).
- test (836 dog images).

There are 33 to 96 dog images in each category with the mean of 63 dog images per category.

The 

[Dogs per category](images/dogspercategory.png)




Detect Humans
Since we want to identify the most resembling dog breed for a person, a function needs to be written to detect whether a human face exists in an image. This project used a pre-trained face detector provided by OpenCV. Please note that the input image is converted to grayscale before it is fed into the face cascade classifier.


Detect Dogs
Similarly, a dog detector function is needed to determine whether there is actually a dog in the input image. A pre-trained ResNet-50 model is used in this project to detect dogs in images.


Keras CNNs require input images to be converted into 4D tensors, so some pre-processing is needed for the image data.


The ResNet50_predict_labels function takes an image path as input, and returns the predicted label of that image using the pre-trained ResNet50 model. The ResNet50 dictionary shows that labels between 151 and 268 are all dogs, therefore the dog_detector function can take advantage of this logic to determine whether the input image contains a dog.


CNN to Classify Dog Breeds using Transfer Learning
The full dataset has 8,351 dog images, which is not large enough to train a deep learning model from scratch. Therefore, transfer learning with VGG-19 ( a convolutional neural network that is trained on more than a million images from the ImageNet database) is used to achieve relatively good accuracy with less training time.

Bottleneck Features
The bottleneck features for the VGG-19 network were pre-computed by Udacity, and then imported for later use by the transfer learning model.


Model Architecture
The last convolutional output of VGG-19 is fed as input to the model. We only need to add a global average pooling layer and fully connected layers as dog classifiers.

I added two fully connected layers for better accuracy, and a dropout layer to prevent over-fitting.


Below is my model architecture:


Model Metric
Accuracy is chosen as the metric to evaluate the model performance. Since data is just slightly imbalanced, accuracy should be a proper metric to select a good model.


Train Model
The model is trained using the pre-computed bottleneck features as input. A model check pointer is used to keep track of the weights for best validation loss. When all epochs are finished, the model weights with the best validation loss are loaded into the VGG19_model, which will be used later for predictions.


Make Predictions
Finally, it is ready to make predictions. The VGG19_predict_breed function takes an image path as input, and returns the predicted dog breeds. The dog_breed_pred function is built on the previous one, and returns predicted results depending on whether a dog or a human is detected in the input image.


Results
The accuracy of the final model on test dataset is about 73%, which is not bad. Originally, I trained a CNN model from scratch without using Transfer Learning, the accuracy was only 1.55%. Then, I created a CNN model using transfer learning and VGG-19 with only one fully connected layer, and was able to reach an accuracy of about 53%. Finally, I added a second fully connected layer to the classifier, and was able to achieve 73% accuracy.

When given an image of a dog, the final model predicts the dog breed. For example,



If a human is in the input image, it identifies the most resembling dog breed based on the person’s face.


Below is a picture of Basenji dog I found online. Does it look somewhat similar to the person above?


Image Source: https://www.akc.org/dog-breeds/basenji/
When the image does not contain a human or a dog, it will tell you that there is no human or dog detected. For example, if I provide a cat picture to the model, it does not try to predict its breed, which is expected.


Conclusion
Thanks to the transfer learning technique, I was able to train a model with relatively small dataset, and achieved pretty good accuracy. In addition, the model was trained within a short period of time, which is quite efficient. The main reason is we can reuse the weights trained by machine learning experts using millions of images.

The initial model was a CNN from scratch, which did not work well. It only reached an accuracy of 1.55%, slight better than random guess. I think it is because the size of dataset is relatively small, and the model architecture might not be well designed.

There are a few possible improvements for the model. First, the parameters of fully connected layers, such as number of layers, number of nodes, dropout percentages, might be tweaked to get better results. Second, using a different optimizer or evaluation metric may also improve model performance. Third, data augmentation could also improve the final model accuracy, as it will generate more training data.

Machine LearningTransfer LearningImage RecognitionData ScienceDeep Learning
Go to the profile of Shuo Wang
Shuo Wang
Related reads
Kuzushiji-MNIST - Japanese Literature Alternative Dataset for Deep Learning Tasks
Go to the profile of Rani Horev
Rani Horev
Dec 14, 2018
Related reads
A Beginner’s Tutorial on Building an AI Image Classifier using PyTorch
Go to the profile of Alexander Wu
Alexander Wu
Feb 4
Related reads
Build a simple Image Retrieval System with an Autoencoder
Go to the profile of Nathan Hubens
Nathan Hubens
Aug 24, 2018
Responses
Conversation with Shuo Wang.
Go to the profile of Niteesh Kanungo
Niteesh Kanungo
Jan 8
Can you please share the weights and the source code link?

Go to the profile of Shuo Wang
Shuo Wang
Jan 9
Hi, this is my github: https://github.com/swang13/dog-breeds-classification





