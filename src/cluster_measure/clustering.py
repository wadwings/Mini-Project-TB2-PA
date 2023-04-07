import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from src.preprocess.setup import config
from src.distance_compute.distance_compute import *

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

def KMeans_clustering(data):
    #process
    return



def do_clustering(data, method = None):
    if method is not None:
        config.set_clustering_method(method)
    clustering_method = config.get_clustering()
    labels = clustering_method(data)
    new_matrix = new_feature_add_matrix(labels, distance_m=)
    return new_matrix


def new_feature_add_matrix(labels, distance_m):

    kmeans = KMeans(n_clusters=k_value, random_state=10).fit(distance_m)
    plt.figure()
    plt.scatter(distance_m[:, 0], distance_m[:, 1], c=kmeans.labels_, cmap='rainbow')

    # Suppose you already have an eigenmatrix X of the shape (m, n)
    # Suppose you have used KMeans to cluster X into k categories

    labels = kmeans.fit_predict(distance_m)
    # Converts the category to which each sample belongs into a binary vector of length k
    print(f'labels: {labels.shape}')
    print(f'distance_m: {distance_m.shape}')
    #Add the binary vector as a new feature to the original feature matrix
    new_clustering_matrix = distance_m
    print(new_clustering_matrix)
    return new_clustering_matrix


if __name__ == '__main__':
    config.set_config(config.speciesType.human, config.chainType.alpha)
    data = load_data().iloc[:200, :]
    data = compute_count(data, config.get_columns())
    distance_matrix = compute_distance(data, data)
    new_feature_add_matrix(2, distance_matrix)




