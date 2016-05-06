# Multi-Class Classification
Automated handwritten recognition is widely used nowadays - from recognizing of zip codes on mail envelopes to recognizing amounts written on bank checks (Is that still a thing?).

There's several techniques that can be used for this classification task. We'll use logistic regression and neural networks to recognize handwritten digits (from 0 to 9).

# Neural Networks
Multi-class Logistic Regression helps us to recognze handwritten digits. However, it cannot form complex hypothesis as it is only a linear classifier.

A Neural Network will be able to represent complex models that form non-linear hypothesis. Through a feedforward propagation algorithm and the parameters of an already trained neural network, we'll be able to predict handwritten digits (in *ex3_nn.m*).

# Dataset
The dataset in _ex3data1.mat_ contains 5000 training examples of handwritten digits in an Octave/Matlab matrix format. This dataset is loaded in memory in _ex3.m_.

Each training example is a 20 pixel by 20 pixel grayscale image of a digit. Each pixel is represented by a floating point number indicating its grayscale intensity.

The 20 by 20 matrix of pixels is unfolded into a 400-dimensional vector, becoming a single row in our data matrix. This gives us a 5000 by 400 matrix where every row is a training example for a handwritten digit image.

### Exercise 3
The second part of the training set is a 5000-dimensional vector that contains labels for the training set. Each label maps the 10 possible digits to a certain value, so digits from **1** to **9** are labeled as **1** to **9**, while **0** is labeled as **10** (we won't use a zero index to make things more compatible with Octave/Matlab indexing).
### Exercise 4
REcall what whereas the original labels were 1, 2, 3, ..., 10, for the purpose of training a neural network, we have to recode the labels as vectors containing only values 0 or 1, so that, for example, if x_i is an image of the digit 5, then the corresponding y_i should be a 10-dimensional vectorwith y_5 equal to 1, and the rest of the elements equal to 0.

# Data visualization
The _displayData_ function maps each row to a 20 pixel by 20 pixel grayscale image and displays all the images together.
![alt text](https://github.com/charlesaraya/neural-networks/blob/master/img/handwritten-digits-displayData.png "Handwritten Digits matrix")

# Files
## Main
_ex3.m_ - Octave/Matlab script that steps through the Multi-Class Classification exercise

*ex3_nn.m* - Octave/Matlab script thet steps through the Neural Network execrise

_ex4.m_ - Octave/Matlab script that steps through the Neural Network Learning exercise

## Data
_ex3data1.mat_ - Training set of Hand-written digits

_ex3weights.mat_ - Trained weights used to test the Neural Network exercise

## Helper functions
_displayData.m_ - Function that helps visualizing the hand-written digits dataset

_fmincg.m_ - Function minimization routine (similar to fminunc)

_computeNumericalGradient.m_ - Numerically compute gradients

_checkNNGradients.m_ - Function to help check your gradients

_debugInitializeWeights.m_ - Function for initializing weights

## Main functions
_sigmoid.m_ - Sigmoid function used in Logistic Regression

_sigmoidGradient.m_ - Compute the gradient of the sigmoid function

_lrCostFunction.m_ - Logistic Regression Cost function

_oneVsAll.m_ - Train a one-vs-all multi-class classifier

_predictOneVsAll.m_ - Predict using one-vs-all multi-class classifier

_predict.m_ - Neural Network prediction function

_randInitializeWeights.m_ - Randomly initialize weights

_nnCostFunction.m_ - Neural network cost fucntion
