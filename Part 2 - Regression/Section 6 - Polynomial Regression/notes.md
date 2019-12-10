# Polynomial linear regression

Formula:

    y = b0 + b1x1 + b2x1^2 + ... + bnx1^n

Depends on data, polynomial curve might fit data better than line
- e.g. disease spread is not simple linear

## Why still linear?

Linear is not about the X variable, but about the coefficient (which is linear).

Coefficients are variables that change, goal is to find them.
Polynomial linear regression is a special case of multiple regression, where coefficients are linear

## Example

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
# X should be a matrix, not a vector
y = dataset.iloc[:, 2].values

# Fitting linear regression model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting polynomial regression model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 5)
# degrees of polynomial features = 2 (... up to x^2)
X_poly = poly_reg.fit_transform(X)

# fit linear regression to polynomial variables
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising linear regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Salary per year of experience (linear regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualise the polynomial regression results
# using X_grid to create an array where step is 0.1, instead of 1
# aim is to have continuos curve, and more precise model

X_grid = np.arange(min(X), max(X), 0.1) # gives a vector
X_grid = X_grid.reshape((len(X_grid), 1)) # reshape into a matrix
plt.scatter(X, y, color = 'red')
# use poly_reg.fit_transform(X) instead of X_poly, easier to change if observations change
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue') # using X_grid instead of X

plt.title('Salary per year of experience (polynomial regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# Predict a new result with linear regression
lin_reg.predict([[6.5]]) # = $330378
lin_reg_2.predict(poly_reg.fit_transform([[6.5]])) # = $174878
```

