# Logistic regression

- predict probability of being in a certain category
  - e.g. 0 or 1

- apply a sigmoid function to linear equation
  - $y = b_{0} + b_{1}*x$
  - $p = \frac{1}{1 + e^{-y}}$
- then solve for y ...

Obtain formula for logistic regression:

$ln(\frac{p}{1-p}) = b_{0} + b_{1}*x$

Best fitting line that fits observations
- can be used to predict probability $\hat{p}$ 
- from 0 to 1
- can pick 0.5 as dividing value for 2 groups

# Example

```python
# Fitting Logistic Regression to training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the test set results
y_pred = classifier.predict(X_test)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising the training set results
from matplotlib.colors import ListedColormap
#create local variables
X_set, y_set = X_train, y_train

#prepare the grid with all pixel points
#taking minimum value of age (-1 so to have more space around the chart), up until max value
#same for salary
#step = 0.01 (otherwise it would not have been dense)
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() +1, step = 0.01), np.arange(start = X_set[:, 1].min() -1, stop = X_set[:, 1].max() + 1, step = 0.01))

#apply classifier to all pixel points
#make contour around 2 predictions
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

#loop to add all data point
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)

#add other info
plt.title('Logistic Regression (training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
```

To visualise plot:
1. taken all pixels of frame, and applied prediction to it
2. colorise  based on prediction
3. added scatter plot for each data point




