function [J grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)

%   NNCOSTFUNCTION Implements the neural network cost function for a two layer
%   neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% Recode y labeling
y_recoded = zeros(m, num_labels);

for i = 1:m
  y_recoded(i, y(i)) = 1;
end;

y = y_recoded;

% Part 1: Feedforward the neural network and return the cost in the variable J.

a1 = [ones(m, 1) X]; % R^m x (input_layer_features)+1 
z2 = a1 * Theta1';
a2 = [ones(m, 1) sigmoid(z2)];  % R^m x (hidden_layer_units)+1
z3 = a2 * Theta2'; 
a3 = sigmoid(z3); % R^m x num_labels(output_layer_labels)
pred = a3; 

Theta1_reg = [ zeros(size(Theta1, 1)) Theta1(:, 2:end)];
Theta2_reg = [ zeros(size(Theta2, 1)) Theta2(:, 2:end)];

J = (1/m) * sum(sum(-y .* log(pred) - (1 - y) .* log(1 - pred))) + ...
    (lambda/(2*m)) * (sum(sum(Theta1_reg.^2)) + sum(sum(Theta2_reg.^2)));

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
% Hint:   We recommend implementing backpropagation using a for-loop
%         over the training examples if you are implementing it for the 
%         first time.

for t = 1:m
  % Feedforward
  a1 = [1 ; X(t, :)'];		% input_layer_size + 1 x 1
  z2 = Theta1 * a1;
  a2 = [1 ; sigmoid(z2)];	% hidden_layer_size + 1 x 1
  z3 = Theta2 * a2;
  a3 = sigmoid(z3);		% output_layer_size x 1
  
  % Backpropagation

  % delta error calculation for the output layer
  d3 = a3 - y(t, :)';	% output_layer_size x 1
  % delta calculation for the hidden layer
  d2 = Theta2' * d3 .* [1;sigmoidGradient(z2)]; % hidden_layer_size x 1
  d2 = d2(2:end);	% remove bias unit :  d2_0
  
  % Accumulate the gradient and then remove d2_0
  Theta1_grad = Theta1_grad + d2 * a1'; % hidden_layer_size x input_layer_size
  Theta2_grad = Theta2_grad + d3 * a2';	% output_layer_size x hidden_layer_size  

end;
% Obtain the (unregularized) gradient for the nn cost function by dividing
% the accumulated gradients by m
Theta1_grad = Theta1_grad/m;
Theta2_grad = Theta2_grad/m;


% Part 3: Implement regularization with the cost function and gradients.
%
% Hint:   You can implement this around the code for
%         backpropagation. That is, you can compute the gradients for
%         the regularization separately and then add them to Theta1_grad
%         and Theta2_grad from Part 2.


% -------------------------------------------------------------

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
