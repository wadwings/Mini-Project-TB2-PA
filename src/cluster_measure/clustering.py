import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
import numpy as np


def check_best_k(distance_m):
    inertias = []
    K = 10
    for k in range(1, K + 1):
        kmm = KMeans(n_clusters=k).fit(distance_m)
        kmm.fit(distance_m)
        inertias.append(kmm.inertia_)
    # Plot the elbow
    plt.figure()
    plt.plot(range(1, K + 1), inertias, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('The elbow method showing the optimal k')


def kmeans_method(k,distance_matrix):
    kmeans = KMeans(n_clusters=k, random_state=10).fit(distance_matrix)
    plt.figure()
    plt.scatter(distance_matrix[:, 0], distance_matrix[:, 1], c=kmeans.labels_, cmap='rainbow')
    return kmeans


def new_feature_add_matrix(label, distance_m):
    kmeans = label
    labels = kmeans.fit_predict(distance_m)
    # Converts the category to which each sample belongs into a binary vector of length k
    encoder = OneHotEncoder(categories='auto')
    onehot_labels = encoder.fit_transform(labels.reshape(-1, 1)).toarray()

    #Add the binary vector as a new feature to the original feature matrix
    new_clustering_matrix = np.hstack([distance_m, onehot_labels])
    return new_clustering_matrix



