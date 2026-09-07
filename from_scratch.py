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

def initialize_centroids(x,k):
   indices = np.random