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
print("Inertia:", inertias)
print("Silhouette Score:", silhouette_scores)
print('callanki_scores:', callanki_scores)
print('davies_scores:', davies_scores)

