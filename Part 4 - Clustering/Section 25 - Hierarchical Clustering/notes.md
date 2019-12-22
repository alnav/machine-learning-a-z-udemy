# Hierarchical clustering

- similar outcome to K-Means cluster, different process

2 types:
- agglomerative (bottom -> up)
- divisive (up -> bottom)

## Agglomerative HC

1. make each data point a single-point cluster -> that forms N clusters
2. take 2 closest **data point**, make them one cluster -> N-1 clusters
3. take 2 closest **clusters** and make them one cluster -> N-2 clusters
4. repeat step 3 until only 1 cluster

## How to measure distance between clusters

Euclidian distance between $P_1$ and $P_2$ = $\sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2 {}}$

Options:

1. distance between closest points
2. distance between furthest points
3. average distance
4. distance between centroids

## Dendogram

Data points vs euclidean distance

- height is euclidian distance
  - distance represents dissimilarity
- horizontal line connects points in cluster
- come up with tree chart
- contains memory of hierarchical cluster algorithm

## How many clusters? - dissimilarity treshold

Can set **dissimilarity (or distance) threshold**

- e.g. can't have a cluster above certain dissimilarity
- count vertical lines below threshold ot know how many clusters

### Optimal # of clusters:
**Find longest vertical line that does not cross any horizontal line.**

# Example

```python
# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
# ward minimise variance between clusters

plt.title('Dendogram')
plt.xlabel('Customer')
plt.ylabel('Euclidean distances')
plt.show()
```