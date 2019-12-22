#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 12:00:09 2019

@author: ale
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])
    
# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# min support, for a product purchased 3 times a day:
# 3*7(days in a week)/7500 (total transactions) = 0.003

# if confidence set too high, 2 products might be associated just because they are very often bought overall
# e.g. people buy loads of mineral water, and also eggs. They will be associated

# Visualising the results
results = list(rules)
