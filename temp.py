import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import scipy.optimize as op

data = pd.read_csv('ex2data1.txt', header=None)
X = data.iloc[:, 0:2].values 
y = data.iloc[:, 2].values[:, None] # -*- coding: utf-8 -*-
X = np.hstack((np.ones((X.shape[0], 1)), X))

def Sigmoid(z):
    return 1/(1 + np.exp(-z));

def Gradient(theta,x,y):
    m , n = x.shape
    theta = theta.reshape((n,1));
    y = y.reshape((m,1))
    sigmoid_x_theta = Sigmoid(x.dot(theta));
    grad = ((x.T).dot(sigmoid_x_theta-y))/m;
    return grad.flatten();

def CostFunc(theta,x,y):
    m,n = x.shape; 
    theta = theta.reshape((n,1));
    y = y.reshape((m,1));
    term1 = np.log(Sigmoid(x.dot(theta)));
    term2 = np.log(1-Sigmoid(x.dot(theta)));
    term1 = term1.reshape((m,1))
    term2 = term2.reshape((m,1))
    term = y * term1 + (1 - y) * term2;
    J = -((np.sum(term))/m);
    return J;



m , n = X.shape;
initial_theta = np.zeros(n);
Result = op.minimize(fun = CostFunc, x0 = initial_theta, args = (X, y),method = 'TNC',jac = Gradient);
optimal_theta = Result.x;