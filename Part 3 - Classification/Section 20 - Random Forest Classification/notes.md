# Random Forest Classification

- Ensamble learning
  - multiple machine learning algorithms work together
- Combines several decision trees

Steps
1. pick a random K data points from training set
2. build decision tree based on these K data points
3. choose number n of trees
   1. repeat step 1 and 2
4. For a new data point, make each tree predict the correct category
5. Assign new data point to category that has majority of votes

Can help get rid of errors made by single algorithm

# Example
```python
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
classifier.fit(X_train, y_train)
```