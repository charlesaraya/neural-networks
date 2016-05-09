function p = predict(Theta1, Theta2, X)
%   PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

m = size(X, 1);
num_labels = size(Theta2, 1);
p = zeros(size(X, 1), 1);

%  We'll feedforward the neural network with test data.
%  It'll give us the activations of the output layer, as a vector of values from 0 to 1.
%  The max function returns the index of the max element of a given vector.
%  This element will be the prediction. 

a1 = [ones(m, 1) X];
a2 = [ones(m, 1) sigmoid(a1 * Theta1')];
a3 = sigmoid(a2 * Theta2');

[max_values p] = max(a3');

% =========================================================================

end
