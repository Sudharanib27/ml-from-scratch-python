#!/usr/bin/env python
# coding: utf-8

# ### Evalvation Model2 : Predict sepal_length based on petal_length and petal_width
# 
# This script performs the following functions:
# 1. Load the Iris dataset from sklearn
# 2. Prepare the input data with petal_length and petal_width as input features and sepal_length as the output feature
# 3. Split the dataset into training and testing sets
# 4. Load the regularized and non regularized trained model parameters
# 5. Calculate the model score and print the mean square error(MSE)

# In[1]:


# loading necessary libraries
from Linear_Regression_Model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[2]:


# Loading IRIS data
iris_dataset = load_iris()
X = iris_dataset.data 
target = iris_dataset.target 
names = iris_dataset.target_names

df = pd.DataFrame(X, columns=iris_dataset.feature_names)
df = df.rename(columns={'sepal length (cm)': 'sepal_length','sepal width (cm)': 'sepal_width','petal length (cm)': 'petal_length','petal width (cm)': 'petal_width'})


# In[3]:


#Preparing the data
X = df[['sepal_length','petal_width']].values
y = df[['petal_length']].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# ### Regularized

# In[4]:


lr_model2 = LinearRegression()

#load the saved trained weights and bias parameters
params = np.load("model2_weights_regularized.npy", allow_pickle=True).item()
weights = params["weights"]
bias = params["bias"]

#load the weights and bias into the model
lr_model2.load_model_parameters(weights, bias)

# Calculate mean squared error
mse = lr_model2.score(X_test, y_test)

# Print the result
print("Mean Squared Error for Model 2:", mse)


# ### Non Regularized

# In[5]:


lr_model2_nonreg = LinearRegression()

# load the saved trained weights and bias parameters
params = np.load("model2_weights_non_regularized.npy", allow_pickle=True).item()
weights = params["weights"]
bias = params["bias"]

#load the weights and bias into the model
lr_model2_nonreg.load_model_parameters(weights, bias)

# Calculate mean squared error
mse = lr_model2_nonreg.score(X_test, y_test)

# Print the result
print("Mean Squared Error for non regularized Model 2:", mse)

