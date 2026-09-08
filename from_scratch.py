import pandas as pd
import numpy as np

df = pd.read_csv('wholesale_data.csv')

print(df.columns)
print(df.info())

features = [
    "Fresh",
    "Milk",
    "Grocery",
    "Frozen",
    "Detergents_Paper",
    "Delicassen"
]
X = df[features]
X = X.to_numpy()
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
x_scalar = (X - mean) / std

np.random.seed(42)
def initialize_centroids(x,k):
   indices = np.random.choice(len(x), size=k, replace=False)
   centroids = []
   for i in indices:
      centroids.append(x[i])
   return centroids

def euclidean_distance(point, centroid):
   d = 0
   for i in range(len(centroid)):
      d += (point[i] - centroid[i])**2
   return np.sqrt(d)

def nearest_distances(point, centroids):
   distances = []
   for centroid in centroids:
      distances.append(euclidean_distance(point, centroid))
   min_len = float('inf')
   for i in range(len(distances)):
      if distances[i] < min_len:
         min_len = distances[i]
         centroid_index = i
   return centroid_index

def cluster_assignments(x, centroids):
   clusters = []
   for point in x:
     clusters.append(nearest_distances(point, centroids))
   return clusters

def calculate_centroid(x,clusters,k):
   centroids = []
   for cluster in range(k):
      sums = [0]*len(x[0])
      count = 0
      for i in range(len(x)):
         if clusters[i] == cluster:
            point = x[i]
            count += 1
            for feature_index in range(len(x[i])):
               sums[feature_index] += x[i][feature_index]

      centroid = []
      for sum in sums:
         centroid.append(sum/count)
      centroids.append(centroid)
   return centroids

def kmeans(x, k):
   centroids = initialize_centroids(x,k)
   for _ in range(100):
      clusters = cluster_assignments(x, centroids)
      new_centroids = calculate_centroid(x,clusters,k)
      if np.allclose(new_centroids,centroids):
         break
      centroids = new_centroids
   return centroids


def inertia(x, clusters, centroids):
    total = 0
    for i in range(len(x)):
       cluster = clusters[i]
       centroid = centroids[cluster]
       for feature_index in range(len(x[i])):
          total += (x[i][feature_index] - centroid[feature_index])**2
    return total
             

for k in range(2, 11):

    best_inertia = float('inf')

    for _ in range(10):

        centroids = kmeans(x_scalar, k)
        clusters = cluster_assignments(x_scalar, centroids)
        current_inertia = inertia(x_scalar, clusters, centroids)

        if current_inertia < best_inertia:
            best_inertia = current_inertia

    print('k:', k, 'inertia:', best_inertia)

 
def average_intra_distance(x, clusters, point_index):
    point = x[point_index]
    cluster = clusters[point_index]
    sum_ = 0
    mean = []
    count = 0
    for i in range(len(x)):
      if clusters[i] == cluster and i != point_index:
            distance = euclidean_distance(point, x[i])
            sum_ += distance
            count += 1
    if count == 0:
       return 0
    return sum_/count

def average_inter_distance(x, clusters, point_index):
       point = x[point_index]
       cluster = clusters[point_index]
       sum_ = 0
       mean = []
       count = 0
       for other_cluster in range(3):
          if other_cluster != cluster:
             sum_ = 0
             count = 0
             for i in range(len(x)):
               if clusters[i] == other_cluster:
                  distance = euclidean_distance(point, x[i])
                  sum_ += distance
                  count += 1
             mean.append(sum_/count)
       return min(mean)

def silhouette_point(x, clusters, point_index):
    a = average_intra_distance(x, clusters, point_index)
    b = average_inter_distance(x, clusters, point_index)
    s = (b - a) / max(a,b)
    return s

def silhouette_score(x, clusters):
   scores = []

   for point_index in range(len(x)):
      scores.append(silhouette_point(x, clusters, point_index))

   return sum(scores) / len(scores)
s_score = silhouette_score(X, clusters)

print('Silhouette Score:', s_score)




    