    # ====================== YOUR CODE HERE ======================
    # Instructions: Perform a single gradient step on the parameter vector
    #               theta. 
    #
    # Hint: While debugging, it can be useful to print out the values
    #       of the cost function (computeCost) and gradient here.
    #
import numpy as np
from computeCost import computeCost
from sigmoid import sigmoid
    
def gradientDescent(X, y, theta, alpha, num_iters):
#GRADIENTDESCENT Performs gradient descent to learn theta
#   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
#   taking num_iters gradient steps with learning rate alpha

# Initialize some useful values
    m = np.size(y, axis = 0) # number of training examples
    J_history = np.zeros((num_iters, 1))
    temp = theta
    for iterator in range(num_iters):
        h = 1 / (1 + np.exp(-(np.dot(theta.T, X.T)).T))
        temp[0,0] = temp[0,0] - alpha*sum(h - y)/m
        temp[1,0] = temp[1,0] - alpha*sum((h - y)*X[:,[1]])/m
        theta = temp
        # Save the cost J in every iteration    
        J_history[iterator] = computeCost(X, y, theta)
    return theta, J_history
