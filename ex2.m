%% Machine Learning Online Class - Exercise 2: Logistic Regression
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the logistic
%  regression exercise. You will need to complete the following functions 
%  in this exericse:
%
%     sigmoid.m
%     costFunction.m
%     predict.m
%     costFunctionReg.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc

%% Load Data
%  The first two columns contains the exam scores and the third column
%  contains the label.

data = load('car.txt');
X = data(:, [1,2,3,4,5,6]); 
y = data(:, 7);
sizeY=size(y);
temp = zeros(sizeY,4);

for i =1:sizeY
	for j=1:size(temp,2) %iteramos sobre las columnas de temp	
		if y(i)==j   % si el dato de la columna es igual al numero de la columna le asignamos el numero de la columna
		      temp(i,j)=1;
		else
		      temp(i,j)=0;
		endif
	endfor
endfor

%% ==================== Part 1: Plotting ====================
%  We start the exercise by first plotting the data to understand the 
%  the problem we are working with.



%% ============ Part 2: Compute Cost and Gradient ============
%  In this part of the exercise, you will implement the cost and gradient
%  for logistic regression. You neeed to complete the code in 
%  costFunction.m

%  Setup the data matrix appropriately, and add ones for the intercept term
[m, n] = size(X);

% Add intercept term to x and X_test
X = [ones(m, 1) X];

% Initialize fitting parameters
initial_theta = zeros(n + 1, 1);
max = -1


%final_thetas = zeros(,size(temp,2));

final_thetas = zeros(size(X,2),size(temp,2));

for i = 1 : size(temp,2)  % uno contra todos.

	% Compute and display initial cost and gradient
	[cost, grad] = costFunction(initial_theta, X, temp(:,i) );
	%fprintf('Cost at initial theta (zeros): %f\n', cost);
	%fprintf('Gradient at initial theta (zeros): \n');
	%fprintf(' %f \n', grad);


	%si clase1 o clase2
	% else y cambiar /eliminar
	%pause;
	%end
	
	%% ============= Part 3: Optimizing using fminunc  =============
	%  In this exercise, you will use a built-in function (fminunc) to find the
	%  optimal parameters theta.
	
	%  Set options for fminunc
	options = optimset('GradObj', 'on', 'MaxIter', 4000);
	
	%  Run fminunc to obtain the optimal theta
	%  This function will return theta and the cost 
	[theta, cost] = ...
		fminunc(@(t)(costFunction(t, X, temp(:,i))), initial_theta, options);
	
	% Print theta to screen
	fprintf('Cost at theta found by fminunc: %f\n', cost);
	fprintf('theta: \n');
	fprintf(' %f \n', theta);

	% Plot Boundary
	%plotDecisionBoundary(theta, X, yTemp);
	
	% Put some labels 
	
	
	%fprintf('\nProgram paused. Press enter to continue.\n');
	
	%pause;
	
	%% ============== Part 4: Predict and Accuracies ==============
	%  After learning the parameters, you'll like to use it to predict the outcomes
	%  on unseen data. In this part, you will use the logistic regression model
	%  to predict the probability that a student with score 45 on exam 1 and 
	%  score 85 on exam 2 will be admitted.
	%
	%  Furthermore, you will compute the training and test set accuracies of 
	%  our model.
	%
	%  Your task is to complete the code in predict.m
	
	%  Predict probability for a student with score 45 on exam 1 
	%  and score 85 on exam 2 
	
	%prob = sigmoid([1 1.1,2.1,3.4,4.4,5.1,6.3] * theta);
	%prob = sigmoid([1 1.4 2.4 3.4 4.2 5.3 6.1] * theta);
	%prob = sigmoid([1 1.4 2.4 3.5 4.4 5.2 6.2] * theta);
	%prob = sigmoid([1 1.4 2.4 3.5 4.6 5.3 6.3] * theta);
	final_thetas (:,i) = theta;
	%prob = sigmoid([1 1.2 2.1 3.3 4.4 5.3 6.2] * theta);
	%fprintf(['Para  i = %d se calculo ' ...
	%        ' una probabilidad de: %f\n\n'], i ,prob);
	
	% Compute accuracy on our training set
	%p = predict(theta, X);
	%rpta =  mean(double(p == temp(:,i))) * 100;
	
	%fprintf('Train Accuracy: %f\n', mean(double(p == temp(:,i))) * 100);
	
	%fprintf('\nProgram paused. Press enter to continue.\n');
	%pause;
endfor

prueba = load('car-prueba.data');
xData = ones(1,size(prueba,2)+1);
xData = prueba(:, [1,2,3,4,5,6]); 
xData = horzcat(ones( size(xData,1),1),xData);

for i = 1 :  size(xData,1)
	
	for j = 1 :  size(final_thetas,2)
		prob = sigmoid(xData(i,:) * final_thetas(:,j));
		fprintf(['Para  el test i = %d probabilidad de clase j = %d es de: %f\n\n'], i, j ,prob);
			
	endfor
	
endfor
