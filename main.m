%% Machine Learning 

%% Initialization
clear ; close all; clc

%% Setup the parameters
input_layer_size  = 400;  % 20x20 Input Images of Digits
num_labels = 10;          % 10 labels, from 1 to 10 (note that we have mapped "0" to label 10)

%% =========== Part 1: Loading and Visualizing Data =============
% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('data.mat'); % training data stored in arrays X, y
m = size(X, 1);

% Randomly select 100 data points to display
rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);

displayData(sel);

fprintf('\nPress enter to continue.\n');
pause;

%% ============ Part 2: Vectorize Logistic Regression ============
%  Implement one-vs-all classification for the handwritten digit dataset.

fprintf('\nTraining One-vs-All Logistic Regression...\n')

lambda = 0.1;
[all_theta] = oneVsAll(X, y, num_labels, lambda);

fprintf('\nPress enter to continue.\n');
pause;

%% ================ Part 3: Predict for One-Vs-All ================

pred = predictOneVsAll(all_theta, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

%% ================ Part 2: Loading Parameters ================
% Load some pre-initialized neural network parameters.

fprintf('\nLoading Saved Neural Network Parameters ...\n')

load('weights.mat'); % Load the weights into variables Theta1 and Theta2
nn_params = [Theta1(:) ; Theta2(:)]; % Unroll parameters 

%% ================ Part 3: Check Cost Function (Feedforward) ================
%  Verify the Cost Function, with and without regularization with the fixed 
%  debugging weights parameters.

fprintf('\n Checking Cost Function without regularization...\n')

lambda = 0; % weight regularization paramater (0 means no regularization)

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);

fprintf(['Cost at parameters (loaded from weights.mat): %f '...
         '\n(this value should be about 0.287629)\n'], J);

fprintf('\nPress enter to continue.\n');
pause;

fprintf('\nChecking Cost Function with Regularization ... \n')

lambda = 1;

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);

fprintf(['Cost at parameters (loaded from ex4weights): %f '...
         '\n(this value should be about 0.383770)\n'], J);

fprintf('\nPress enter to continue.\n');
pause;

%% ================ Part 3: Sigmoid Gradient  ================
%  For large values (positive and negative) of z, the gradient should be close to 0.
%  When z = 0, the gradient should be exactly 0.25.
%  We'll use the sigmoid gradient to compute the hidden layer delta error.

fprintf('\nEvaluating sigmoid gradient ...\n')

g = sigmoidGradient([1 -0.5 0 0.5 1]);
fprintf('\nSigmoid Gradient evaluated at [1 -0.5 0 0.5 1]:...\n');
fprintf('%f ', g);

fprintf('\nPress enter to continue.\n');
pause;

%% ================ Part 6: Initializing Pameters ================
%  Initialize the weights of the neural network

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%% =============== Part 7: Check Backpropagation ===============
%  For the backpropagation algorithm for the neural network we need to
%  check gradients by running checkNNGradients.

fprintf('\nChecking Backpropagation without regularization... \n');

checkNNGradients;

fprintf('\nPress enter to continue.\n');
pause;

fprintf('\nChecking Backpropagation with regularization ... \n')

lambda = 3;
checkNNGradients(lambda);

% Also output the costFunction debugging values
debug_J  = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);

fprintf(['\n\nCost at (fixed) debugging parameters (w/ lambda = 10): %f ' ...
         '\n(this value should be about 0.576051)\n\n'], debug_J);

fprintf('\nPress enter to continue.\n');
pause;

%% =================== Part 8: Training the Neural Network ===================
%  To train the NN we'll use "fmincg"; it works similarly to "fminunc". 
%  Recall that these advanced optimizers are able to train our cost functions 
%  efficiently as long as we provide them with the gradient computations.
%
fprintf('\nTraining Neural Network... \n')

options = optimset('MaxIter', 50); % Change the MaxIter to a larger value to see how more training helps.
lambda = 1; %  Try different values of lambda

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);

% costFunction is a function that takes in only one argument (the neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('\nPress enter to continue.\n');
pause;

%% ================= Part 9: Visualize Weights =================
%  We can now "visualize" what the neural network is learning by displaying the 
%  hidden units to see what features they are capturing in 

fprintf('\nVisualizing Neural Network... \n')

displayData(Theta1(:, 2:end));

fprintf('\nPress enter to continue.\n');
pause;

%% ================= Part 10: Implement Predict =================
%  After training the neural network, we'll predict the labels. 

pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);