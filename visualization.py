import matplotlib.pyplot as plt

k_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]
inertias = [1954, 1642, 1318, 1070, 971, 836, 764, 720, 617]

plt.plot(k_values, inertias, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.savefig('elbow.png')
plt.show()