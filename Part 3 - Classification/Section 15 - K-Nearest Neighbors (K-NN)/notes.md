# K-Nearest neighbors

When adding each new data point, follow steps:
1. Choose number K of neighbors (default = 5)
2. Take the K nearest neighbors of the new data point, according to euclidean distance (most often)
3. Among these K neigbors, count number of data point in each category
4. Assign new data point to category where you counted the most neigbors

Euclidean distance is geometrical distance (hypotenuse)

$d = \sqrt{(x_{2} - x_{1})^2+(y_{2} - y_{1})^2}$

```python
# Fitting K neighbors classifier to training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2) # for euclidean distance
classifier.fit(X_train, y_train)