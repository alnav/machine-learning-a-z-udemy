# Evaluating regression models performance

## R-Squared

- sum of square of residuals
  - $SSres = SUM (y_{i} - \hat{y})^2$
- total sum of squares
  - $SStot = SUM (y_{i} - y_{average})^2$

**R-squared**
- $R^2 = 1 - \frac{SSres}{SStot}$

**Trying to fit a line to minimise SSres**
- because SStot will always have a value, there will always be an average of values
- fitting a slope line that minimise SSres, best line, e.g. by OLS

**$R^2$ tells us how good is that line**
- goes from 1 (best, when SSres is very low) to negative
- goodness of fit

# Adjusted R-squared

- Problem with R-squared:
  - when adding variables, R-squared will never decrease
  - new variable either helps or worsen the process of minimising $SS_{res}$
  - if worsen, will be given a very small coefficient and made irrelevant
  - but there there will always be a random correlation between new variable in respect to dependent variable
    - if variable has absolutely no correlation, model will find a random (even if small) correlation, and $R^2$ will go up

**Formula:**

$Adj R^2 = 1 - (1 - R^2)\frac{n - 1}{n - p - 1}$

- p = number of regressors
- n = sample size
- penalised for adding independent variables that don't help
  - when adding p, denominator decrease, ratio increase, so 1 - ... decrease and total value goes down
  - at the same time, $R^2$ goes up when adding variables
  - there is a balance between adding variables that only benefit the model

Adjusted R-squared has a penalisation factor as you add more variables

# Interpreting coefficients

|                 | Estimate  | Standard Error | T-value | P-value |
| --------------- | --------- | -------------- | ------- | ------- |
| R.D.Spend       | 7.966e-01 | 4.135e-02      | 19.266  | <2e-16  |
| Marketing spend | 2.991e-02 | 1.552e-02      | 1.927   | 0.06    |


1. look at sign of estimate: if positive, independent variable is correlated with dependent variable
2. magnitude can be tricky:
   1. can vary based on scale
   2. cannot comment on a variable having greater importance
   3. think about change in dependent variable, *PER UNIT CHANGE* of independent variable
   4. e.g. per dollar spend in R.D Spend, increase in profit by 79 cents


