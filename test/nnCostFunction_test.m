il = 2;              % input layer
hl = 2;              % hidden layer
nl = 4;              % number of labels
nn = [ 1:18 ] / 10;  % nn_params
X = cos([1 2 ; 3 4 ; 5 6]);
y = [4; 2; 3];
lambda = 4;
[J grad] = nnCostFunction(nn, il, hl, nl, X, y, lambda)

% With Regularization
%J = 19.474

% Without Regularization
%J =  7.4070

%grad =
%0.76614
%0.97990
%0.37246
%0.49749
%0.64174
%0.74614
%0.88342
%0.56876
%0.58467
%0.59814
%1.92598
%1.94462
%1.98965
%2.17855
%2.47834
%2.50225
%2.52644
%2.72233

% unregularized gradient deltas prior scalation (1/m)

%Delta1 = 
%  2.298 -0.082 -0.074
%  2.939 -0.107 -0.161

%Delta2 =
%  2.650  1.377  1.435
%  1.706  1.033  1.106
%  1.754  0.768  0.779
%  1.794  0.935  0.966
