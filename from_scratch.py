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

def eucludean_distance(point, centroid):
   d = 0
   for i in range(len(centroid)):
      d += (point[i] - centroid[i])**2
   return np.sqrt(d)

def nearest_distances(point, centroids):
   distances = []
   for centroid in centroids:
      distances.append(eucludean_distance(point, centroid))
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

def kmeans(x, k=3):
   centroids = initialize_centroids(x,k)
   for _ in range(100):
      clusters = cluster_assignments(x, centroids)
      new_centroids = calculate_centroid(x,clusters,k)
      if np.allclose(new_centroids,centroids):
         break
      centroids = new_centroids
   return centroids
print('done')
print(kmeans(x_scalar))






