function [J, grad] = lrCostFunction(theta, X, y, lambda)
%   LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%   regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize variables
J = 0;
grad = zeros(size(theta));

m = length(y); % number of training examples

% Compute the hypothesis
pred = sigmoid(X * theta);

temp_theta = theta;
temp_theta(1) = 0;

J = (1/m) * sum(-y .* log(pred) - (1 - y) .* log(1 - pred)) + lambda/(2*m) * sum(temp_theta .^ 2);

grad = (1/m) * X' * (pred - y) + (lambda/m) * temp_theta;

% =============================================================

grad = grad(:);

end
