#!/usr/bin/env python
# coding: utf-8

# ## Multi output Linear Regression:
# 
# This script performs the following functions:
# 1. Load the Iris dataset from sklearn
# 2. Prepare the input data with sepal_length, sepal_width as input features and petal_length, petal_width as the output feature
# 3. Split the dataset into training and testing sets
# 4. Train a linear regression model with regularization value
# 5. Plot the MSE loss graph
# 6. Calculate the model score and print the mean square error(MSE)

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

#Preparing the data
df = pd.DataFrame(X, columns=iris_dataset.feature_names)
df = df.rename(columns={'sepal length (cm)': 'sepal_length','sepal width (cm)': 'sepal_width','petal length (cm)': 'petal_length','petal width (cm)': 'petal_width'})
X = df[['sepal_length','sepal_width']].values
y = df[['petal_length','petal_width']].values

#splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# In[3]:


# initializing and training the model
lr_model = LinearRegression(max_epochs = 100, learning_rate = 0.01, regularization=0.1)
lr_model.fit( X_train, y_train)

#Plot the loss graph
plt.plot(range(len(lr_model.mse_loss_history)), np.ravel(lr_model.mse_loss_history), label='Training Loss')
plt.xlabel('Step Number')
plt.ylabel('Loss')
plt.title('Training Loss Over Steps')
plt.legend()
plt.show()


# In[4]:


# calculating model performance
mse = lr_model.score(X_test, y_test)

# Print the result
print("Mean Squared Error for Model 4:", mse)


# In[5]:


print("predicted value", np.round(lr_model.y_pred,1))


# In[6]:


print("Actual value", y_test)

