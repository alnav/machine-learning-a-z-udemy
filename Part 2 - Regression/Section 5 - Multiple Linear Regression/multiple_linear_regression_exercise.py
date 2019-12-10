#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 18:55:10 2019

@author: ale
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding dummy variable trap
X = X[:, 1:]

# Feature scaling
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

# Backward elimination
import statsmodels.api as sm

# statsmodel library does not take into account b0 constant
# need to add column of 1 in matrix of features

X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
# added column of 1s to matrix X, can also use statsmodels.tools.tools.add_constant(array)

# STEP 1: select significance level
# SL = 0.05

# STEP 2: fit the model with all possible predictors
# Create optimal matrix of features
X_opt = X[:,[0, 1, 2, 3, 4, 5]] # contains all independent variables

# create new regressor from statsmodel
# exog .. intercept is not included by default, needs to be added by the user (code at line 45)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() # ordinary least square

# STEP 3: consider the predictor with the highest P-value. if P > SL, go to step 4, otherwise FINISH
regressor_OLS.summary()

# in this case, it's x2
# STEP 4: remove the predictor
X_opt = X[:,[0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# repeat steps 3 and 4
X_opt = X[:,[0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# repeat steps 3 and 4
X_opt = X[:,[0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# p-value for x2 is 0.06, almost below 0.05 but still higher, will remove

X_opt = X[:,[0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# R&D spend is the only significative variable in predicting profit

