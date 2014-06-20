function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
% J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
% parameter for logistic regression and the gradient of the cost
% w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
% You should set J to the cost.

% Initialize some useful values
n = size(X,2); % number of features + 1 (for constant term)

h = zeros(m,1);


for i = 1:m
    %theta
    %X(i,:)
    d = X(i,:) * theta;
    %h(i) = 1.0/(1.0 + exp(-d));
    h(i)=sigmoid(d);
    %fprintf('d=%f, h(%d)=%f\n', d, i, h(i));
end

J = (-1.0/m)*sum(y.*log(h) + (1-y).*log(1-h));

for j = 1:n
    d = h-y;
    v = d'*X(:,j);
    grad(j) = (1.0/m)*sum(v);
end 
% =============================================================

end