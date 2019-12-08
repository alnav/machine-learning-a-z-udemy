# Preprocessing data

- independent variables
  - variable analysed by our algorithms
- dependent variables
  - answer or outcome

## Importing libraries
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```


## Importing dataset
```python
dataset = pd.read_csv('data.csv')
```
- create matrix of features
```python
X = dataset.iloc[:, :-1].values
  # [first is lines, second is columns]
- Y = dataset.iloc[:, -1]
```
## Missing data
- there will be missing data in your dataset
- ways to handle:
  - remove data point entirely
    - dangerous, what if there is crucial info?
  - take the mean of the column
```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0)
imputer = imputer.fit(X[:, 1:3]) # upper bound is excluded

X[:, 1:3] = imputer.transform(X[:, 1:3])
```
   
- Imputer is deprecated
  - use from sklearn.impute import SimpleImputer instead

- in python, [:, 1:3] second number is upper bound, in this case it selects column 1 and 2


## Categorical variables
- run into trouble if we keep string into equations
- need to encode text into numbers
  - both country and purchased column
- ISSUE: if we just encode into 0, 1 or 2, there will be different weight into the equation
  - use OneHotEncoder with ColumnTransformer to have dummy variables
```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
  
labelEncoder_X = LabelEncoder()
X[:,0] = labelEncoder_X.fit_transform(X[:,0])
  
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
  
X = np.array(ct.fit_transform(X), dtype=np.float)
  
labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)
```

## Split dataset into training set and test set
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```

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
```python
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) 
```

