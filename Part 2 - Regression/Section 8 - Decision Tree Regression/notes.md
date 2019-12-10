# Decision trees

CART = Classification and regression trees
- regression trees are more complicated

- algorithm splits data into segments (called leaves)
  - e.g. x1 > 20
    - then x2 > 170 etc...
- based on information entropy
  - is the split increasing amount of information? is it adding value?
    - or too small leaf (e.g. <5% of sample)

## Predict y
- for new observation, y = average(values_inside_leaf)

# Example

```python
# Fitting the Regression Model to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
# default criterion = MSE (mean squared error)
regressor.fit(X, y)
```

- looks too good on first look
  - line connects all data points, because of low resolution
  - if increase levels (step 0.1), then chart becomes several hard steps, not continuous
- decision tree regression model is not continuous
  - sometimes not interesting in 2D, but can be very useful with multiple dimensions

