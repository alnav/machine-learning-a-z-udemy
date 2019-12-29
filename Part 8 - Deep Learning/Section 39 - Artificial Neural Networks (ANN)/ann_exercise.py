#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 11:08:24 2019

@author: ale
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Data preprocessing
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X_1 = LabelEncoder()
X[:, 1] = labelEncoder_X_1.fit_transform(X[:, 1])

labelEncoder_X_2 = LabelEncoder()
X[:, 2] = labelEncoder_X_2.fit_transform(X[:, 2])

from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [1])], 
                        remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)

X = X[:, 1:] # avoid the dummy variable trap

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Create the ANN

# Import Keras and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialise neural network
classifier = Sequential()

# Adding input layer and first hidden layer 
# choose rectifier activation function for hidden layer, sigmoid function for output
# how many hidden nodes? (take average(node in input layer and output)) - units = 6 (11+2)/2
# relu is rectifier
# needs input_dim on first layer
classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform", input_dim = 11))

# add 2nd hidden layer
classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))

# Adding the output layer
# if dealing with several categories, will need multiple output neurons, and activation function would be softmax
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))

# Compiling the ANN
# adam is stochastic gradient descent algorithm
# use logarithmic loss function with sigmoid function (binary_crossentropy with 1 category)
# metric expect analysing parameters in a list, e.g. accuract
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fit ANN to training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# threshold to change y_pred from probability to true/false
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

