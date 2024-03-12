#!/usr/bin/env python
# coding: utf-8

# ## Input to Logistic Regression Model:  All features
# 
# This script performs the following functions:
# 1. Load the Iris dataset from sklearn
# 2. Prepare the input data with petal length, petal width, sepal length and sepal width
# 3. Split the dataset into training and testing sets
# 4. Standardize the features
# 5. Train a logistic regression model
# 6. Predict test values using trained logistic model
# 7. Calculate accuracy of the model
# 8. Print confusion matrix
# 9. Plot decision regions using mlxtend

# In[1]:


# Import necessary libraries
from Logistic_Regression_Model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# In[2]:


# Loading iris data
iris_dataset = load_iris()
X = iris_dataset.data
# Converting to binary classification problem (0 = setosa, 1 = not setosa)
y = iris_dataset.target

#splitting the input data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# scaling input and output data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[3]:


# training logistic regression model
lgrg_model = LogisticRegression()
lgrg_model.fit(X_train_scaled, y_train)

# Predicting classes
y_pred = lgrg_model.predict(X_test_scaled)

# Calculating accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print accuracy
print(f"Accuracy on test set with petal features: {accuracy}\n")
print("predicted values: \n", y_pred)
print("actual values: \n", y_test)
# Printing confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
print('confusion_matrix:\n', cnf_matrix)

