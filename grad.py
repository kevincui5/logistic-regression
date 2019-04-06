# -*- coding: utf-8 -*-
import numpy as np
from sigmoid import sigmoid

def gradient(theta, X, y):
    if theta.ndim == 1:
        theta = theta[:, None]
    m = np.size(y, axis=0) # number of training examples
    z = np.dot(X, theta)
    h = sigmoid(z)
    g = h - y
    grad = (g * X).sum(axis = 0) / m
    return grad.flatten()

def gradient2(theta,x,y):
    m , n = x.shape
    theta = theta.reshape((n,1));
    y = y.reshape((m,1))
    sigmoid_x_theta = sigmoid(x.dot(theta));
    grad = ((x.T).dot(sigmoid_x_theta-y))/m;
    return grad.flatten();