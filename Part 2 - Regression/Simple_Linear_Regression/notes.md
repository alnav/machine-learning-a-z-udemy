# Simple linear regression

## Formula
y = b0 + b1*x1
- y is dependend variable (outcome)
- x is independent variable, only one in simple linear regression
  - trying to find out association between y and x
- b1 coefficient, define how a unit change in x1 affect unit change in y
  - slope of the line
  - kind of multiplier, connects two values
  - x1 is often not directly proportional to y
- b0 is constant term
  - value of y if x = 0

## Example
- salary depends on employee experience
- salary = b0 + b1*experience
- try to create a best fitting line
- b0 will be salary at x = 0 (just started work, no experience)

## Ordinary least squares
- difference between y (observation) and y^ (modeled)
- SUM (y - y^)^2 -> minimum
- find a line that has the minimum sum of squares
  - will be the best fitting line
  - called ordinary least squares method

## How to fit linear regressor to training set
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

## Predicting test set results
    y_pred = regressor.predict(X_test)
- y_pred are predicted salaries (by our model)
- y_test contains real salaries
- useful to compare the 2

## Visualising training set results
    
    plt.scatter(X_train, y_train, color = 'red')
    plt.plot(X_train, regressor.predict(X_train), color = 'blue') 
    # using X_train as Y line
    plt.title('Salary vs Experience (training set)')
    plt.xlabel('Years of experience')
    plt.ylabel('Salary')
    plt.show()

    plt.scatter(X_test, y_test, color = 'red')
    plt.plot(X_train, regressor.predict(X_train), color = 'blue')
    plt.title('Salary vs Experience (test set)')
    plt.xlabel('Years of experience')
    plt.ylabel('Salary')
    plt.show()

