from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
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

model = KMeans(init = 'random', n_clusters=3, n_init=10,random_state=42)

model.fit(x_scalar)
print(model.predict(x_scalar))