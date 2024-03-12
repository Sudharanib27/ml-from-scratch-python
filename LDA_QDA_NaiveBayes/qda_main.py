#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Solution for custom QDA on the CIFAR-10 dataset.

Author: Sudharani Bannengala
"""

from sklearn.metrics import accuracy_score
from utils import load_and_prepare_data
from models import QDAModel

def rgb_calculate():
    print("Loading data...", end="")
    # Load and prepare the data
    train_data, train_labels, test_data, test_labels = load_and_prepare_data()
    print("done.")

    # Reshape the data
    train_data = train_data.reshape(len(train_data), -1)
    test_data = test_data.reshape(len(test_data), -1)

    # Create and fit the QDA model
    qda = QDAModel()
    print("Fitting QDA model...", end="")
    qda.fit(train_data, train_labels)
    print("done.")

    # Predict the test set labels
    test_preds = qda.predict(test_data)

    # Calculate test set accuracies
    test_acc = accuracy_score(test_labels, test_preds)

    print(f"Test accuracy: {test_acc}")


def grayscale_calculate():
    print("Loading data...", end="")
    # Load and prepare the data
    train_data, train_labels, test_data, test_labels = load_and_prepare_data(True)
    print("done.")

    # Reshape the data
    train_data = train_data.reshape(len(train_data), -1)
    test_data = test_data.reshape(len(test_data), -1)

    # Create and fit the QDA model
    qda = QDAModel()
    print("Fitting QDA model...", end="")
    qda.fit(train_data, train_labels)
    print("done.")

    # Predict the test set labels
    test_preds = qda.predict(test_data)

    # Calculate test set accuracies
    test_acc = accuracy_score(test_labels, test_preds)

    print(f"Test accuracy: {test_acc}")


def main():
    print("****************************************")
    print("*        RGB Custom QDA Solution       *")
    print("****************************************")
    rgb_calculate()

    print("\n****************************************")
    print("*     Grayscale Custom QDA Solution    *")
    print("****************************************")
    grayscale_calculate()


if __name__ == "__main__":
    main()

