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
    labels = []
    # process
    return labels


def do_clustering(data, method=None):
    if method is not None:
        config.set_clustering_method(method)
    clustering_method = select_method()
    labels = clustering_method(data)
    new_matrix = append_clustering_result(labels, distance_m=data)
    return labels, new_matrix


def select_method(method=None):
    if method is None:
        method = config.cluster_method
    return {
        config.clusterMethodType.KMeans: KMeans_clustering,
        'default': KMeans_clustering,
    }.get(method, 'default')


def append_clustering_result(labels, distance_m):
    new_clustering_matrix = distance_m
    # merge matrix here
    return new_clustering_matrix


if __name__ == '__main__':
    config.set_config(config.speciesType.human, config.chainType.alpha)
    data = load_data().iloc[:200, :]
    data = compute_count(data, config.get_columns())
    distance_matrix = compute_distance(data, data)

    append_clustering_result(2, distance_matrix)
