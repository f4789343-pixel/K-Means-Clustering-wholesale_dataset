from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import pandas as pd

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

scalar = StandardScaler()
x_scalar = scalar.fit_transform(X)

k_values = []
inertias = []
silhouette_scores = []
callanki_scores = []
davies_scores = []
for k in range(2, 11):
   model = KMeans(init = 'random', n_clusters=k, n_init=10,random_state=42)

   model.fit(x_scalar)
   clusters = model.predict(x_scalar)

   inertia = model.inertia_
   s_score = silhouette_score(x_scalar, clusters)
   c_score = calinski_harabasz_score(x_scalar, clusters)
   d_score = davies_bouldin_score(x_scalar, clusters)

   k_values.append(k)
   inertias.append(inertia)
   silhouette_scores.append(s_score)
   callanki_scores.append(c_score)
   davies_scores.append(d_score)
best_inertia_k = k_values[inertias.index(min(inertias))]
best_silhouette_k = k_values[silhouette_scores.index(max(silhouette_scores))]
best_calinski_k = k_values[callanki_scores.index(max(callanki_scores))]
best_davies_k = k_values[davies_scores.index(min(davies_scores))]

print("Best k by Inertia:", best_inertia_k,min(inertias))
print("Best k by Silhouette:", best_silhouette_k, max(silhouette_scores))
print("Best k by Calinski-Harabasz:", best_calinski_k, max(callanki_scores))
print("Best k by Davies-Bouldin:", best_davies_k, min(davies_scores))

