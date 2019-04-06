import numpy as np
from sigmoid import sigmoid

# doesn't give correct optimal theta result.  couldn't find reason
def costFunction(theta, X, y):
#COSTFUNCTION Compute cost and gradient for logistic regression
#   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
#   parameter for logistic regression and the gradient of the cost
#   w.r.t. to the parameters.

# Initialize some useful values
    m = np.size(y, axis=0) # number of training examples
    z = np.dot(X, theta)
    h = sigmoid(z)

# You need to return the following variables correctly 
    J = np.sum(-y * np.log(h) - (1 - y) * np.log(1 - h)) / m


# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta.
#               You should set J to the cost.
#               Compute the partial derivatives and set grad to the partial
#               derivatives of the cost w.r.t. each parameter in theta
#
# Note: grad should have the same dimensions as theta
#

    return J

def costFunction2(theta,x,y):
    m,n = x.shape; 
    theta = theta.reshape((n,1));
    y = y.reshape((m,1));
    term1 = np.log(sigmoid(x.dot(theta)));
    term2 = np.log(1-sigmoid(x.dot(theta)));
    term1 = term1.reshape((m,1))
    term2 = term2.reshape((m,1))
    term = y * term1 + (1 - y) * term2;
    J = -((np.sum(term))/m);
    return J;