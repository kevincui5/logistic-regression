from pylab import scatter, show, legend, xlabel, ylabel
from numpy import where

def plotData(X, y):
#PLOTDATA Plots the data points X and y into a new figure 
#   PLOTDATA(x,y) plots the data points with + for the positive examples
#   and o for the negative examples. X is assumed to be a Mx2 matrix.
    pos = where(y == 1)
    neg = where(y == 0)
    scatter(X[pos, 0], X[pos, 1], marker='x', c='b')
    scatter(X[neg, 0], X[neg, 1], marker='o', c='y')
    xlabel('Exam 1 score')
    ylabel('Exam 2 score')
    legend(['Admitted', 'Not Admitted'])
    show()

