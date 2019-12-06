#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 19:47:13 2019

@author: ale
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values # on lecture is was [:, :-1].values, not supported anymore
y = dataset.iloc[:, -1] # or [:, 3]

# Deal with missing data
from sklearn.impute import SimpleImputer
#from sklearn.preprocessing import Imputer #deprecated
#imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
# strategy = mean and axis = 0 are default
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0)
imputer = imputer.fit(X[:, 1:3]) # upper bound is excluded

X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:,0] = labelEncoder_X.fit_transform(X[:,0])
# 0 = France, 1 = Germany, 2 = Spain
# won't work in an equation, need to use a dummy variable
# need 3 columns, one for each category
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], 
                        remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)

#DEPRECATED
#oneHotEncoder = OneHotEncoder(categorical_features = [0])
# categorical features specify which column to encode
#X = oneHotEncoder.fit_transform(X).toarray()

# for Y, no need for OneHotEncoder as only 2 categories
labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)
# 0 is no, 1 is yes

# Split dataset into training set and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, 
                                                    random_state = 0)

# Feature scaling
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) 
# no need to fit, it's already fitted to training set, 
# important to do this way so they are scaled similarly

# no need to apply feature scaling to dependent variable if classification between 2 





