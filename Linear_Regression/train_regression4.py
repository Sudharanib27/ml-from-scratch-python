#!/usr/bin/env python
# coding: utf-8

# ### Training Model4 : Predict petal_length based on sepal_length and petal_width
# 
# This script performs the following functions:
# 1. Load the Iris dataset from sklearn
# 2. Prepare the input data with sepal_length and petal_width as input features and petal_length as the output feature
# 3. Plot the data distribution graph
# 4. Split the dataset into training and testing sets
# 5. Train a linear regression model with no regularization value
# 6. Train a linear regression model with regularization value
# 7. Plot the MSE loss graph
# 8. Print weight parameter values of regularized and non regularized models

# In[1]:


from Linear_Regression_Model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[2]:


# Loading IRIS data
iris_dataset = load_iris()
X = iris_dataset.data 
target = iris_dataset.target 
names = iris_dataset.target_names

#preparing input dataframe
df = pd.DataFrame(X, columns=iris_dataset.feature_names)
df = df.rename(columns={'sepal length (cm)': 'sepal_length','sepal width (cm)': 'sepal_width','petal length (cm)': 'petal_length','petal width (cm)': 'petal_width'})


# In[3]:


#Plotting the data distribution plot between input features
sns.scatterplot(data=df, x="sepal_length", y="petal_width",hue ='petal_length')
plt.title('Data distribution')
plt.show()


# In[4]:


# Preparing the data for training
X = df[['sepal_length','petal_width']].values
y = df[['petal_length']].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# ### Non regularized training:

# In[5]:


lr_model4_non_reg = LinearRegression(max_epochs = 100, learning_rate = 0.01, regularization=0)
lr_model4_non_reg.fit( X_train, y_train)

#saving model weights and bias
lr_model4_non_reg.save_model_parameters("model4_weights_non_regularized.npy")

#Plot the loss graph
plt.plot(range(len(lr_model4_non_reg.mse_loss_history)), np.ravel(lr_model4_non_reg.mse_loss_history), label='Training Loss')
plt.xlabel('Step Number')
plt.ylabel('Loss')
plt.title('Training Loss Over Steps')
plt.legend()
plt.show()


# ### Regularized training:

# In[6]:


lr_model4_reg = LinearRegression(max_epochs = 100, learning_rate = 0.01,regularization=0.1)
lr_model4_reg.fit( X_train, y_train)

#saving model weights and bias
lr_model4_reg.save_model_parameters("model4_weights_regularized.npy")

#Plot the loss graph
plt.plot(range(len(lr_model4_reg.mse_loss_history)), np.ravel(lr_model4_reg.mse_loss_history), label='Training Loss')
plt.xlabel('Step Number')
plt.ylabel('Loss')
plt.title('Training Loss Over Steps')
plt.legend()
plt.show()


# In[7]:


#Analyzing the difference between the weights between regularized and non regularized models
weight_difference = lr_model4_reg.weights - lr_model4_non_reg.weights
print('non regularized weights value : ',lr_model4_non_reg.weights)
print('regularized weights value     : ',lr_model4_reg.weights)
print('Difference in weight          :',weight_difference)


#  
