import matplotlib.pyplot as plt
from scikit_learn import x_scalar, clusters, model

centroids = model.cluster_centers_
plt.scatter(x_scalar[:,0], x_scalar[:,1], c=clusters)
plt.scatter(centroids[:,0], centroids[:,1], marker='X', s=200)
plt.xlabel('Fresh products')
plt.ylabel('Milk')
plt.title('K-Means Clustering')
plt.savefig('fresh_vs_milk.png')
plt.show()

centroids = model.cluster_centers_
plt.scatter(x_scalar[:,2], x_scalar[:,3], c=clusters)
plt.scatter(centroids[:,2], centroids[:,3], marker='X', s=200)
plt.xlabel('Grocery')
plt.ylabel('Frozen')
plt.title('K-Means Clustering')
plt.savefig('grocery_vs_frozen.png')
plt.show()

k_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]
inertias = [1954, 1642, 1318, 1070, 971, 836, 764, 720, 617]

plt.plot(k_values, inertias, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.savefig('elbow.png')
plt.show()

k_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]
Silhouette_Score = [0.5532526964184259, 0.33391714199926514, 0.38314465075241005, 0.37162792634848146, 0.35009963092551916, 0.26950668553505414, 0.30438644818579963, 0.2985429349406454, 0.28803377315030293]

plt.plot(k_values, Silhouette_Score, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.savefig('silhouette_score.png')
plt.show()