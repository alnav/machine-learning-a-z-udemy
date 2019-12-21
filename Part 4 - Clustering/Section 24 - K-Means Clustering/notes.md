# K-Means Clustering

- Allows to discover categories or groups into a dataset

## Steps

1. choose the number K of clusters
2. select at random K points, centroids of cluster 
   - not necessarily from dataset, can be any value
3. assign each data point to closest (e.g. euclidian distance) centroid, that forms K clusters
4. compute and place the new centroid of each cluster made from step 3
5. reassign each data point to new closest centroid
6. if any reassignment, repeat from step 4

## Random initialisation trap

- Cluster can be different, based on initial centroids
- selection of initial centroids can change outcome

K-means++ algorithm can help correct:
- no need to implement, works in the background

## WCSS - within cluster sum of squares

$WCSS = \sum distance(Pi, C1)^2 + \sum distance(Pi, C2)^2 + \sum distance(Pi, C3)^2{}{}$

- sum per each cluster
  - for each point in a cluster: sum of $distance(P,Cn)^2$
- can have a max of clusters = data points
  - WCSS will be 0, each points has each own centroid

To find optimal number of clusters: **elbow method**
- visually when drop is not big anymore

# Example

```python
# Using the elbow method to find optimal number of clusters
from sklearn.cluster import KMeans

wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

```