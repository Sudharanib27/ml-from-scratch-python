#!/usr/bin/env python
# coding: utf-8

# # Linear Regression with single target output

# ### Note:
# This is a linear regression model designed to perform regression tasks. It accepts input features X and single output target values y as parameters. The model is initialized with optional parameters such as the learning rate and the number of iterations for gradient descent optimization.

# In[1]:


#importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.model_selection import train_test_split


# In[2]:


class LinearRegression:
    def __init__(self, batch_size=32, regularization=0, max_epochs=100, patience=3, learning_rate=0.01, validation_size=0.1):
        """Linear Regression using Gradient Descent.

        Parameters:
        -----------
        batch_size: int
            The number of samples per batch.
        regularization: float
            The regularization parameter.
        max_epochs: int
            The maximum number of epochs.
        patience: int
            The number of epochs to wait before stopping if the validation loss
            does not improve.
        learning_rate: float
            The step size at which the model parameters are updated during training.
        validation_size: float
            The sample size selected for validation
            
        """
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
        self.learning_rate = learning_rate
        self.validation_size = validation_size
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """Fit a linear model.

        Parameters:
        -----------
        X: numpy.ndarray
            Training input features.
        y: numpy.ndarray
            Training output values.
        
        """

        # Initialize the weights and bias based on the shape of X and y.
        self.X = X
        self.y = y
        self.samples_count, self.features_count = self.X.shape
        self.weights = np.random.rand(self.features_count,1)
        self.bias = 0
        

        # Intitializing the parameters for the training loop
        best_pred_mse_loss = float('inf')
        best_weights = np.copy(self.weights)
        best_bias = self.bias
        increase_count = 0
        
        # saving loss values in array
        self.mse_loss_history = [] 
        
        # splitting the data
        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=self.validation_size, random_state=42)
        
        # Implement the training loop.
        for _ in range(self.max_epochs):
            for i in range(0, len(X_train), self.batch_size):
                X_batch = np.array(X_train[i:i + self.batch_size])
                y_batch = np.array(y_train[i:i + self.batch_size])
                
                y_pred = self.predict(X_batch)
                mse_loss = np.mean((y_pred - y_batch) ** 2) + ((self.regularization / 2) * np.sum(self.weights**2))
                #print('mse_loss',mse_loss)
                #print('regularization',((self.regularization / 2) * (self.weights.T @ self.weights)))
                
                self.mse_loss_history.append(mse_loss)
                
                #calculating gradients
                dW = - ( 2 * ( X_batch.T ).dot( y_batch - y_pred ) ) / len(X_batch) + self.regularization * self.weights
                db = - 2 * np.sum( y_batch - y_pred ) / len(X_batch)
                
                #Updating the values of weights and bias
                self.weights = self.weights - self.learning_rate * dW
                self.bias = self.bias - self.learning_rate * db
        

    def predict(self, X):
        """Predict using the linear model.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        """
        # Implement the prediction function.
        return X.dot( self.weights ) + self.bias

    def score(self, X, y):
        """Evaluate the linear model using the mean squared error.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        y: numpy.ndarray
            The target data.
        """
        # Implement the scoring function.
        y_pred = self.predict(X)
        self.y_pred = y_pred
        n = len(y)  # Number of samples
        m = 1 if len(y.shape) == 1 else y.shape[1]  # Output size
        
        #calculate mean square error
        mse = np.mean((y - y_pred) ** 2) / (n * m)
        return mse
    
    def save_model_parameters(self, filename):
        """saving the trained model output parameters.

        Parameters
        ----------
        filename: string
            name of the file to which weights and bias values to be saved
        """
        np.save(filename, {"weights": self.weights, "bias": self.bias}, allow_pickle=True)
        

    def load_model_parameters(self, weights, bias):
        """saving the trained model output parameters.

        Parameters
        ----------
        weights: float
            The trained model weight value
        bias: float
            The trained model bias value
        """
        self.weights = weights
        self.bias = bias

