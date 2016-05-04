# Multi-Class Classification
Automated handwritten recognition is widely used nowadays - from recognizing of zip codes on mail envelopes to recognizing amounts written on bank checks (Is that still a thing?).

There's several techniques that can be used for this classification task. We'll use logistic regression and neural networks to recognize handwritten digits (from 0 to 9).

## Main
_ex3.m_ - Octave/Matlab script that steps through the Multi-Class Classification exercise
_ex3nn.m_ - Octave/Matlab script thet steps through the Neural Network execrise
## Data
_ex3data1.mat_ - Training set of Hand-written digits
_ex3weights.mat_ - Trained weights used to test the Neural Network exercise
## Helper functions
_displayData.m_ - Function that helps visualizing the hand-written digits dataset
_fmincg.m_ - Function minimization routine (similar to fminunc)
# Main functions
_sigmoid.m_ - Sigmoid function used in Logistic Regression
_lrCostFunction.m_ - Logistic Regression Cost function
_oneVsAll.m_ - Train a one-vs-all multi-class classifier
_predictOneVsAll.m_ - Predict using one-vs-all multi-class classifier
_predict.m_ - Neural Network prediction function
