# Support vector regression

- support linear and nonlinear regression
- instead of trying to fit largest possible "street" between 2 classes while limiting margin violation, SVR tries to fit as many instances as possible on the street while limiting margin violation
- **width** of the street is controlled by hyperparameter *epsilon*

## Suppor vector machines
- SVM perform linear regression in higher dimensional space
- separate categories by a clear gap that is as wide as possible
- new data points are mapped into that space, into a category based on which side of the gap they fall

## Kernel trick
- SVM can perform non-linear classification by implicitly mapping inputs into high-dimensional feature spaces

## Requirements for SVR
- training set which covers the domain of interest and is accompanied by solutions on that domain
  - work of the SVM is to approximate function we used to generate training set: F(x) = y
- vectors X are used to define hyperplane that separate 2 classes in solution
- vectors closed to test point are referred as support vectors

## Building a SVR
1. collect training set
2. choose a kernel and its parameters, as well as any regularisation needed
3. form the correlation matrix, K
4. train machine, exactly or approximately, to get contraction coefficients a = {a}
5. use those coefficients to create estimator: f(X,a,x) = y

### Choose a kernel
- Gaussian (as you move away to training data, machine return min value of training data)

### Regularisation
- reduces noise

## Goal

- in SVR goal is for error not to exceed the threshold
  - in linear regression was to minimise error

# Example
```python
# SVR requires features scaling!
# Fitting the SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
# Kernel options
# rbf = radial basis function
regressor.fit(X, y)

# CEO is considered outlier, won't fit well

# Predicting a new result
# need to transform the data
value_pred = sc_X.transform(np.array([[6.5]]))
y_pred = regressor.predict(value_pred)
y_scalar = sc_y.inverse_transform(y_pred)  # scale back using y scaling
```