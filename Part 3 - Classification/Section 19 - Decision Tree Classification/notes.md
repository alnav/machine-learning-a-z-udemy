# Decision tree - Classification

CART

- classification and regression tree
- classification trees work with categorical data
- regression predict real numbers

How it works:
1. splits data in different groups based on features
   - done so to maximise single category in each group, or minimise entropy
2. assign new points based on features, compare to several steps down the tree

- it was an old method, but revamped recently
- can be used in random forest and gradient boosting

# Example

```python
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy')
classifier.fit(X_train, y_train)
```

