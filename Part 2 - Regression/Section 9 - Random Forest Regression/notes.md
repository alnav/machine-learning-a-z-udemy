# Random forest regression

Ensamble learning
- multiple algorithms, or same multiple times, working together

Steps:
1. pick random K data points from training set
2. build decision tree associated to these K data points
3. choose number of trees to build, and repeat step 1 and 2
4. for a new data point, make each tree predict the value y
5. assign the new data point a y value, which is the average across all predicted y values

- uses average of many predictions

# Example

```python
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)
```

