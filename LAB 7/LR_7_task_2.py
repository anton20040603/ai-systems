# LR_7_task_2.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import pairwise_distances_argmin
import matplotlib
matplotlib.use('TkAgg')

# Завантаження датасету Iris
iris = load_iris()
X = iris['data']    # ознаки: довжина/ширина чашолистка та пелюстки
y = iris['target']  # реальні мітки класів (для порівняння)

# --- Базова кластеризація KMeans (3 кластери = 3 види ірису) ---
kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# Візуалізація: перші дві ознаки (довжина і ширина чашолистка)
plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.title('KMeans clusterization Iris (3 clusters)')
plt.xlabel('Length of sepal')
plt.ylabel('Width of sepal')
plt.show()

# --- Власна реалізація алгоритму k-середніх ---
def find_clusters(X, n_clusters, rseed=2):
    # Ініціалізація: випадково вибираємо n_clusters точок як початкові центри
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]

    while True:
        # Призначення кожної точки до найближчого центроїду
        labels = pairwise_distances_argmin(X, centers)

        # Перерахунок центрів як середніх значень точок кластеру
        new_centers = np.array([X[labels == i].mean(0)
                                for i in range(n_clusters)])

        # Збіжність: якщо центри не змістилися – алгоритм завершений
        if np.all(centers == new_centers):
            break
        centers = new_centers

    return centers, labels

# Кластеризація з rseed=2
centers, labels = find_clusters(X, 3)
plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.title('Own implementation of KMeans (rseed=2)')
plt.xlabel('Length of sepal')
plt.ylabel('Width of sepal')
plt.show()

# Кластеризація з rseed=0 – інший початковий стан
centers, labels = find_clusters(X, 3, rseed=0)
plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.title('Own implementation of KMeans (rseed=0)')
plt.xlabel('Length of sepal')
plt.ylabel('Width of sepal')
plt.show()

# Кластеризація через fit_predict з фіксованим random_state
labels = KMeans(3, random_state=0, n_init=10).fit_predict(X)
plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.title('KMeans fit_predict (random_state=0)')
plt.xlabel('Length of sepal')
plt.ylabel('Width of sepal')
plt.show()
