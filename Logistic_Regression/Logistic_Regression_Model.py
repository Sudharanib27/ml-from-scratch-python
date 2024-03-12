#!/usr/bin/env python
# coding: utf-8

# ## Multi Class Logistic Regression

# In[1]:


# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt


# ### Note:
# This is a "Multi Class Logistic Regression Classifier" which can take input X, y parameters along with a optional parameter of threshold. Fit method will train the data using gradient descent and softmax functions and predict method will provide the decision output based on the trained model. Loss will be calculated using cross entropy loss function.

# In[2]:


class LogisticRegression:
    def __init__(self, learning_rate = 0.01, iterations=1000):
        """Linear Regression using Gradient Descent.

        Parameters:
        -----------
        learning_rate: float
            The learning rate for gradient descent.
        num_iterations: int
            The number of iterations for gradient descent.
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.loss_history = []

    def fit(self, X, y):
        """Fit a linear model.
        
        Parameters:
        -----------
        X: numpy.ndarray
            Training input features.
        y: numpy.ndarray
            Training output values.
        
        """
        self.X = X
        self.y = y
        
        # initialize the parameters
        n, dimension = self.X.shape
        n_classes = len(np.unique(self.y))
        self.weights = np.random.rand(dimension, n_classes)
        self.bias = np.zeros((1, n_classes))

        for _ in range(self.iterations):
            #computing prediction
            y_pred_cal = np.dot(self.X, self.weights) + self.bias
            y_pred = self.softmax(y_pred_cal)
            
            #gradient computation
            dw = (1/n) * np.dot(self.X.T, (y_pred - np.eye(n_classes)[self.y]))
            db = (1/n) * np.sum(y_pred - np.eye(n_classes)[self.y], axis=0, keepdims=True)
            
            #adjust the weights and bias based on the calculated gradient
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            loss = self.compute_entropy_loss(np.eye(n_classes)[self.y], y_pred)
            self.loss_history.append(loss)
            
    def predict(self, X):
        """Predict using the linear model.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        threshold: float
            The value used to decide the decision boundary
        """
        y_pred_cal = np.dot(X, self.weights) + self.bias
        y_pred = self.softmax(y_pred_cal)
        return np.argmax(y_pred, axis=1)
    
    def softmax(self, z):
        """Apply softmax function.

        Parameters
        ----------
        z: numpy.ndarray
            The predicted y value.
        """
        e_x = np.exp(z - np.max(z, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)
    
    def compute_entropy_loss(self, y, y_cal):
        """Compute cross entropy loss function.

        Parameters
        ----------
        y: numpy.ndarray
            The output value providing during training.
        y_cal: numpy.ndarray
            The predicted output value.
        """
        n = y.shape[0]
        loss = -(1/n) * np.sum(y * np.log(y_cal+ 1e-8))
        return loss

