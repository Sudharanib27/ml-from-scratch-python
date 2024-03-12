#!/usr/bin/env python
# coding: utf-8

# In[3]:


# %load models.py
"""
Fit and Predict methods to calculate LDA, QDA and Naive Bayes machiene learning models

Author: Sudharani Bannengala
"""

import numpy as np

class LDAModel:
    def __init__(self):
        """
        Description:
            Initializer method.
        """
        self.class_means = None
        self.shared_cov_matrix = None
        self.priors = None
        self.w = None
        self.b = None
        
    def fit(self, X, y):
        """Fit LDA model.

        Parameters:
        -----------
        X: numpy.ndarray
            Training input features, shape (n_samples, n_dim).
        y: numpy.ndarray
            Training output values, shape (n_samples,).
        
        """
        # Check if input dimensions are matching
        if X.shape[0] != len(y):
            raise ValueError("Number of samples in X and y is not matching")
            
        n_samples, n_dim = X.shape[0], X.shape[1]
        labels = np.unique(y)
        class_count = len(labels)
        
        #initialize the matrices
        self.class_means = np.zeros((class_count, n_dim))
        self.shared_cov_matrix = np.zeros((n_dim, n_dim))
        self.priors = np.zeros(class_count)
        
        # Loop to calculate class means and shared covariance matrix
        for i, label in enumerate(labels):
            X_k = X[y == label]
            #calculate the class means
            self.class_means[i] = np.mean(X_k, axis=0)
            #calculate the class prior values
            self.priors[i] = X_k.shape[0] / n_samples
            #calculate the shared covarance matrix
            cov_k = np.cov(X_k, rowvar=False)
            self.shared_cov_matrix += (X_k.shape[0] - 1) * cov_k
            
        self.shared_cov_matrix /= (n_samples - class_count)
        
        # calculate weight and bias values based on the calculated shared_cov_matrix and class_means
        inv_shared_cov_matrix = np.linalg.inv(self.shared_cov_matrix)
        self.w = np.dot(inv_shared_cov_matrix, self.class_means.T)
        self.b = -0.5 * np.diag(np.dot(np.dot(self.class_means, inv_shared_cov_matrix), self.class_means.T))  + np.log(self.priors)
        
        return self
        
    def predict(self, X):
        """Predict using the LDA model.

        Parameters
        ----------
        X: numpy.ndarray, shape (n_samples, n_dim)
                Training input features.
        
        Returns: shape (n_samples)
             Predicted class labels
        """
        if self.w is None or self.b is None:
            raise ValueError("Model is not fitted properly. Please rerun the fit the model before proceeding")
        
        #calculate the y_pred based on trained values of weight and bias
        y_pred = np.dot(X, self.w) + self.b
        
        return np.argmax(y_pred, axis=1)
                
              
class QDAModel:
    def __init__(self):
        """
        Description:
            Initializer method.
        """
        self.class_means = None
        self.cov_matrices = None
        self.priors = None
        self.labels = None
        
    def fit(self, X, y):
        """Fit a QDA model.

        Parameters:
        -----------
        X: numpy.ndarray
            Training input features, shape (n_samples, n_dim).
        y: numpy.ndarray
            Training output values, shape (n_samples,).
        
        """
        # Check if input dimensions are matching
        if X.shape[0] != len(y):
            raise ValueError("Number of samples in X and y is not matching")
            
        n_samples, n_dim = X.shape[0], X.shape[1]
        self.labels = np.unique(y)
        class_count = len(self.labels)
        
        #initialize the matrices
        self.class_means = np.zeros((class_count, n_dim))
        self.cov_matrices = [np.zeros((n_dim, n_dim)) for _ in range(class_count)]
        self.priors = np.zeros(class_count)
        
        # Loop to calculate class means, class covariance matrices and class prior values
        for i, label in enumerate(self.labels):
            X_k = X[y == label]
            #print(X_k.shape)
            self.class_means[i, :] = np.mean(X_k, axis=0)
            self.priors[i] = X_k.shape[0] / n_samples
            self.cov_matrices[i] = np.cov(X_k.T)
                 
        return self
                  

    def predict(self, X):
        """Predict using the QDA model.

        Parameters
        ----------
        X: numpy.ndarray, shape (n_samples, n_dim)
                Training input features.
        
        Returns: shape (n_samples)
             Predicted class labels
        """
        if self.class_means is None or self.cov_matrices is None:
            raise ValueError("Model is not fitted properly. Please rerun the fit the model before proceeding")
        
        y_pred = np.zeros((X.shape[0], len(self.labels)))
        
        #calculate the y_pred based on trained values of means, priors and variances of each class
        for i in range(len(self.labels)): 
            inv_cov_mat = np.linalg.inv(self.cov_matrices[i])
            sign, log_cov_mat = np.linalg.slogdet(self.cov_matrices[i] )
            log_prior = np.log(self.priors[i])   
            y_pred[:,i] = log_prior - 0.5 * sign * log_cov_mat - 0.5 * np.sum((X - self.class_means[i]) @ inv_cov_mat * (X - self.class_means[i]), axis=1)
        
        return self.labels[np.argmax(y_pred, axis=1)]


class GaussianNBModel:
    def __init__(self):
        """
        Description:
            Initializer method.
        """
        self.class_means = None
        self.class_variances = None
        self.priors = None
        
    def fit(self, X, y):
        """Fit Naive Bayes model.

        Parameters:
        -----------
        X: numpy.ndarray
            Training input features, shape (n_samples, n_dim).
        y: numpy.ndarray
            Training output values, shape (n_samples,).
        
        """
        # Check if input dimensions are matching
        if X.shape[0] != len(y):
            raise ValueError("Number of samples in X and y is not matching")
            
        n_samples, n_dim = X.shape[0], X.shape[1]
        self.labels = np.unique(y)
        class_count = len(self.labels)
        
        #initialize the matrices
        self.class_means = np.zeros((class_count, n_dim))
        self.class_variances = np.zeros((n_dim, n_dim))
        self.priors = np.zeros(class_count)
        
        # Loop to calculate class means and class variances
        for i, label in enumerate(self.labels):
            X_label = X[y == label]
            self.class_means[i, :] = X_label.mean(axis=0)
            self.class_variances[i, :] = X_label.var(axis=0)
            self.priors[i] = len(X_label) / n_samples
        
        return self        

    def predict(self, X):
        """Predict using the Naive Bayes model.

        Parameters
        ----------
        X: numpy.ndarray, shape (n_samples, n_dim)
                Training input features.
        
        Returns: shape (n_samples)
             Predicted class labels
        """
        if self.class_means is None or self.class_variances is None:
            raise ValueError("Model is not fitted properly. Please rerun the fit the model before proceeding")
        
        y_pred = []
        for x in X:
            post_val = []
            # loop for each class to calculate y_pred
            for i in range(len(self.labels)):
                likelihood_inter = -0.5 * np.sum(np.log(2 * np.pi * self.class_variances[i]))
                likelihood_inter -= 0.5 * np.sum(((x - self.class_means[i]) ** 2) / (self.class_variances[i]))
                likelihood = self.priors[i] + likelihood_inter
                post_val.append(likelihood)
                
            y_pred.append(self.labels[np.argmax(post_val)])
            
        return y_pred
    

