# Suppor vector machine

Starts from already agreed observations.

Need to find separation line between groups, boundaries.

- optimal line that divides groups
- trying to maximise margins
- two points that are on the lines, are called support vectors
  - in multidimensional space a point (in 2D) becomes a vector
- line in the middle is called maximum margin hyperplane
  - or maximum margin classifier

- Finds difference in samples that are from different categories, but are the most similar to each other, to find margin

## Example

```python
from sklearn.SVM import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
# default kernel = rbf
classifier.fit(X_train, y_train)
```