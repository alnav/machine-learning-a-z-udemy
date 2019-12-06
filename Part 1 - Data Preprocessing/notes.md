# Preprocessing data


- independent variables
  - variable analysed from our algorithms
- dependent variables
  - answer

## Importing libraries
- essential libraries:
  - import numpy as np
  - import matplotlib.pyplot as plt
  - import pandas as pd
  
## Importing dataset
- dataset = pd.read_csv('data.csv')

- create matrix of features
  - X = dataset.iloc[:, :-1].values.. does not include last column 
    - [left is lines, right is columns]
    - last version of spyder does not use .values anymore
  - Y = dataset.iloc[:, -1]

## Missing data
- there will be missing data in your dataset
- ways to handle:
  - remove data point entirely
    - dangerous, what if there is crucial info?
  - take the mean of the column
    - from sklearn.preprocessing import Imputer
    - imputer = Imputer(missing_values = 'NaN', strategy = 'mean')
      - mean and axis = 0 are default, no need to input
    - imputer = imputer.fix(X[:, 1:3])
- Imputer is deprecated
  - use from sklearn.impute import SimpleImputer instead


- in python, [:, 1:3] second number is upper bound, in this case it selects column 1 and 2


## Categorical variables
- run into trouble if we keep string into equations
- need to encode text into numbers
  - both country and purchased column
- ISSUE: if we just encode into 0, 1 or 2, there will be different weight into the equation
  - use OneHotEncoder with ColumnTransformer to have dummy variables

## Split dataset into training set and test set
- from sklearn.model_selection import train_test_split
- X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
  - usual size of test set is 20-25% of dataset

## Feature scaling
- without scaling, variable will have very different weights
- machine learning based on euclidian distance of different values
  - high value will dominate calculation, and be weighted much more
- ways to scale:
  - standardisation
    - x - mean(x) / SD
  - normalisation
    - x - min(x) / max(x) - min(x)
- fit and trasform training and test set

- need to scale dummy variable?
  - depends on context, how much we need to keep interpretation in the project
- no need to apply feature scaling to dependent variable if classification between 2 

