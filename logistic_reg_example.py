import sys
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
import matplotlib.pyplot as plt
import matplotlib as matplot
import numpy as np
from scipy.optimize import fmin_bfgs
from sklearn import linear_model

class IrisDatasetLogisticRegression:
        """
         Implementing logistic regression using Iris dataset
        """     
        
        """Global class variables"""
        "Matrix containing set of features"
        X = None
        
        "Matrix containing set of outputs"
        y = None
        
        def __init__(self, X, y):
            """ USAGE:
            Default constructor            
           
            PARAMETERS:
            X - feature matrix
            y - output matrix     
            
            RETURN:            
            """          
            self.X = X
            self.y = y            
            
            """Convert y into a proper 2 dimensional array/matrix. This is to facilitate proper matrix arithmetics."""
            if len(y.shape) == 1:
                y.shape = (y.shape[0],1)
            
                    
        def sigmoid(self,z):
            """ USAGE:
            Compute the sigmoid of each value of z (z can be a matrix, vector or scalar).            
           
            PARAMETERS:
            z - Matrix, vector or scalar       
            
            RETURN:
            The sigmoid value
            """    
            return 1.0 / (1.0 + np.exp(-z))
        
        
        def compute_cost(self,X, y, theta):
            """ USAGE:
            Define the cost function           
          
            PARAMETERS:
            X - Features
            y - Output
            theta        
           
            RETURN:
            return the cost function value
            """    
            m = X.shape[0]
            z = np.dot(X,theta)            
            h = self.sigmoid(z);    
           
            J=(float(-1)/m)*((y.T.dot(np.log(h))) + ((1 - y.T).dot(np.log(1 - h))))           
            return J
        
        
        def compute_gradient(self,X, y, theta):
            """ USAGE:                  
            Compute the gradient using vectorization.
            
            PARAMETERS:           
            X - Features
            y - Output
            theta 
            
            RETURN:           
            """    
            m = X.shape[0]
            z = np.dot(X,theta)            
            h = self.sigmoid(z);
    
            grad = (float(1)/m)*((h-y).T.dot(X))          
            return grad
        

        def plot_two_features(self):
            """ USAGE:
            Plot first two features from the Iris dataset  
          
            PARAMETERS:           
            
            RETURN:    
            """      
            fig = plt.figure()
            ax = fig.add_subplot(111, title ="Iris Dataset - Plotting two features", xlabel = 'Sepal Length', ylabel='Sepal Width')
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
                     
                        
            setosa = np.where(self.y == 0)
            versicolour = np.where(self.y ==1)
            
            ax.scatter(X[setosa, 0], X[setosa, 1], s=20, c='r', marker ='o')  
            ax.scatter(X[versicolour, 0], X[versicolour, 1], s=20, c='r', marker ='x') 
            plt.legend( ('Iris Type - Setosa', 'Iris Type - Versicolour') )
            plt.show()
            

        def plot_three_features(self):
            """ USAGE:
            Plot first two features from the Iris dataset  
          
            PARAMETERS:           
            
            RETURN:    
            """      
            fig = plt.figure()
            ax = fig.add_subplot(111, title ="Iris Dataset - Plotting three features", xlabel = 'Sepal Length', ylabel='Sepal Width', zlabel='Petal Length', projection='3d')
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            plt.setp(ax.get_zticklabels(), visible=False)
           
            ax.scatter(X[:, 0], X[:, 1],X[:, 2], s=20, c='r', marker ='o')                   
            plt.show()
            
            
        def run_logistic_regression(self):
            """ USAGE:
            Apply principles of logistic regression
          
            PARAMETERS:           
            
            RETURN:    
            """  
            
            """m= number of training data, n= number of features"""
            m = self.X.shape[0]
            n = self.X.shape[1]
            
            """Add intercept term to X"""
            self.X = np.hstack((np.ones((m, 1)), self.X))
            
            
            """Initialize fitting parameters. Take into account the intercept term."""
            initial_theta = np.zeros((n + 1, 1))
            
            """"Compute initial cost and gradient"""
            cost = self.compute_cost(self.X, self.y, initial_theta)            
            gradient = self.compute_gradient(self.X, self.y, initial_theta)
            
            print ('Cost at initial theta (zeros): {0} \nGradient at initial theta (zeros): {1}'.format(cost, gradient) )
        
            def f(theta):
                return np.ndarray.flatten(self.compute_cost(self.X, self.y, initial_theta))
            
            def fprime(theta):
                return np.ndarray.flatten(self.compute_gradient(self.X, self.y, initial_theta))
            
            print(fmin_bfgs(f, initial_theta, fprime, disp=True, maxiter=400, full_output = True, retall=True))        
          
'''           
try:
#    iris = datasets.load_iris()
#    X = iris.data
#    Y = iris.target
    
    data = np.loadtxt('data/ex2data1.txt', delimiter=',')
    X = data[:, 0:2]   
    Y = data[:, 2]
   
    
    logistic_regression  = IrisDatasetLogisticRegression(X, Y)
    #logistic_regression.plot_two_features()
    #logistic_regression.plot_three_features()
    
    logistic_regression.run_logistic_regression()
    
except:
    print "unexpected error:", sys.exc_info()
'''