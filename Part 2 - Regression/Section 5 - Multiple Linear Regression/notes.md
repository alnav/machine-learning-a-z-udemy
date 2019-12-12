# Multiple linear regression

    y = b0 + b1x1 + b2x2 + b3x3 + ... + bnxn

- b0 = constant
- bn = coefficients
- xn = independent variables

## Assumptions for multiple linear regression
1. linearity
2. homoscedasticity
3. multivariate normality
4. independence of errors
5. lack of multicollinearity

## Dummy variables

Categorical variables don't fit into equation, need to create dummy variable

- Create one column for each category
- 1 if data point has that category, otherwise 0
- if 0, bx = b*0 = 0, does not affect equation
- default equation contains one of 2 dummy variable, information included in b0
- bad idea to include all dummy variables

## Dummy variable trap
- **Multicollinearity**: one or several independent variables in a linear regression predict another
- model can't distinguish between effect of 2 variable, won't work
- can't have constant and all dummy variables

```python
X = X[:, 1:] # remove one column
```

### Always omit one dummy variable, it will be contained in the constant (default state)

## P-value
- p-value is NOT the probability the claim is true
- not the probability that the null hypotesis is true

P-value is probability of getting same results, or more extreme IF the null hypotesis is true.
- How weird is it to have got this result?
- If very high, then not weird at all, we can accept the null hypotesis

Very small p-value indicates that by pure luck, it would be unlikely to get a sample like the one we have, if the null hypotesis is true. It means we can reject the null hypotesis

## Buidling a model
- garbage in, garbage out
  - if all variables are kept, data might not be optimal
- having a thousand variables, it's not practical to interpret model

Several methods
1. All-in
2. Backward elimination
3. Forward selection
4. Bidiretional elimination
5. Score comparison

2-4 are stepwise regression (mostly people refer to bidirectional elimination)

## All-in cases
- if prior knowledge
  - from domain knowledge or given 
- if you have to
  - forced to use a framework, in certain industries
- if you are preparing for backward elimination

## Backward elimination

Several steps:
1. select a significance level to stay in the model
   - Default SL = 0.005 (significance level)
2. fit the full model with all possible predictors
3. consider the predictor with the highest P-value. if P > SL, go to step 4, otherwise FINISH
4. Revove that predictor
5. fit the model without this variable
6. repeat step 3 until variable with highest P-value, is still less than SL

## Forward selection

More complex than reversing backward elimination
1. select significance level (SL) eg. 0.05
2. fit all possible regression models y - Xn, select one with lowest P-value
3. keep this variable, and fit all other possible models with one extra predictor added to model
4. consider the predictor with lowest P-value. If P < SL, go to step 3, otherwise FINISH

## Bidirectional elimination

Combines the two, can be called stepwise regression:
1. Select a SL to enter, and a SL to stay
2. perform next step of forward selection (new variable must have P < SL-ENTER)
3. perform all steps of backward elimination (old variables must have P < SL-STAY)
4. Until no new variable can enter, or old variable can exit, then FINISH

## All possible models or score comparison
1. Select criterion of goodness (e.g. Akaike criterion)
2. costruct all possible regression models
3. select one with best criterion, FINISH

More resource consuming! with 10 columns, have 1023 models! Grows exponentially


## Practicale example
- no need to apply feature scaling, library does it

```python
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
```

- have to find which variables are statistically significant
  - do that by applying backward selection
- statsmodel need b0 = 1 (first column of ones)

```python
import statsmodels.formula.api as sm

# statsmodel library does not take into account b0 constant
# need to add column of 1 in matrix of features

X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
# added column of 1s to matrix X
# added column of 1s to matrix X, can also use statsmodels.tools.tools.add_constant(array)

# STEP 1: select significance level
# SL = 0.05

# STEP 2: fit the model with all possible predictors
# Create optimal matrix of features
X_opt = X[:,[0, 1, 2, 3, 4, 5]] # contains all independent variables

# create new regressor from statsmodel
# exog .. intercept is not included by default, needs to be added by the user (code at line 45)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() # ordinary least square

regressor_OLS.summary()
```


- x2 has P-value of 0.990, needs to be removed

```python
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
```

- continue with algorithm:
```python
# repeat steps 3 and 4
X_opt = X[:,[0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# repeat steps 3 and 4
X_opt = X[:,[0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# p-value for x2 is 0.06, almost below 0.05 but still higher, will remove
X_opt = X[:,[0, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
```
**R&D spend is the only significative variable in predicting profit**

## Automatic backward elimination in python
```python
# with P-values only
import statsmodels.formula.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)

# with P-values and adjusted R-squared

import statsmodels.formula.api as sm
def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)
```