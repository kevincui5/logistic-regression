## Logistic Regression




## Load Data
#  The first two columns contains the exam scores and the third column
#  contains the label.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

data = pd.read_csv('ex2data1.txt', header=None)
X = data.iloc[:, 0:2].values 
y = data.iloc[:, 2].values[:, None] 

print(['Plotting data with + indicating (y = 1) examples and o '
         'indicating (y = 0) examples.\n'])

from plotData import plotData
plotData(X, y)


#Compute Cost and Gradient
from costFunction import costFunction, costFunction2
from grad import gradient, gradient2
#  Setup the data matrix appropriately, and add ones for the intercept term
m, n = np.shape(X)

# Add intercept term to x and X_test
X_nobias = X # for later use
X = np.hstack((np.ones((X.shape[0], 1)), X))

# Initialize fitting parameters
initial_theta = np.zeros((n + 1, 1))

# Compute and display initial cost and gradient
cost = costFunction(initial_theta, X, y)
grad = gradient2(initial_theta, X, y)
print('Cost at initial theta (zeros): %f\n', cost)
print('Cost2 at initial theta (zeros): %f\n', costFunction2(initial_theta, X, y))
print('Expected cost (approx): 0.693\n')
print('Gradient at initial theta (zeros): \n')
print(' %f \n', grad)
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n')

# Compute and display cost and gradient with non-zero theta
test_theta = np.array([[-24], [0.2], [0.2]])
cost = costFunction(test_theta, X, y)
grad = gradient2(test_theta, X, y)
print('\nCost at test theta: %f\n', cost)
print('\nCost2 at test theta: %f\n', costFunction2(test_theta, X, y))
print('Expected cost (approx): 0.218\n')
print('Gradient at test theta: \n')
print(' %f \n', grad)
print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n')

# find the optimal parameters theta.
import scipy.optimize as op
m , n = X.shape;
initial_theta = np.zeros(n);
Result = op.minimize(fun = costFunction, x0 = initial_theta, args = (X, y),method = 'TNC',jac = gradient);
optimal_theta = Result.x;
# Print theta to screen
print('Cost at theta found by fminunc: %f\n', cost)
print('Expected cost (approx): 0.203\n')
print('theta: \n')
print(' %f \n', optimal_theta)
print('theta2: \n')
print(' %f \n', op.minimize(fun = costFunction2, x0 = initial_theta, args = (X, y),method = 'TNC',jac = gradient).x)
print('Expected theta (approx):\n')
print(' -25.161\n 0.206\n 0.201\n')
'''
# Plot Boundary
plotDecisionBoundary(theta, X, y)

# Put some labels 
hold on
# Labels and Legend
xlabel('Exam 1 score')
ylabel('Exam 2 score')

# Specified in plot order
legend('Admitted', 'Not admitted')
'''
## ============== Part 4: Predict and Accuracies ==============
#  After learning the parameters, you'll like to use it to predict the outcomes
#  on unseen data. In this part, you will use the logistic regression model
#  to predict the probability that a student with score 45 on exam 1 and 
#  score 85 on exam 2 will be admitted.
#
#  Furthermore, you will compute the training and test set accuracies of 
#  our model.
#
#  Your task is to complete the code in predict.m

#  Predict probability for a student with score 45 on exam 1 
#  and score 85 on exam 2 

prob = sigmoid([1 45 85] * theta)
print(['For a student with scores 45 and 85, we predict an admission ' ...
         'probability of %f\n'], prob)
print('Expected value: 0.775 +/- 0.002\n\n')

# Compute accuracy on our training set
p = predict(theta, X)

print('Train Accuracy: %f\n', mean(double(p == y)) * 100)
print('Expected accuracy (approx): 89.0\n')
