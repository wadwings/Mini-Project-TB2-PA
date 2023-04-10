import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from src.preprocess.setup import config
from src.distance_compute.distance_compute import *
from sklearn.metrics import silhouette_score
from src.features_extraction.features_extraction import do_features_extraction


def check_best_k(data):
    k_range = range(2, 11)
    silhouette_scores = []
    print(data)
    for k in k_range:
        # 训练K-Means模型
        t = data
        model = KMeans(n_clusters=k, random_state=42)
        model.fit(t)
        score = silhouette_score(t, model.labels_)
        silhouette_scores.append(score)
    # 找到最佳的K值
    best_k = k_range[silhouette_scores.index(max(silhouette_scores))]
    return best_k


def KMeans_clustering(data):
    k = check_best_k(data)
    kmeans = KMeans(n_clusters=k, random_state=10).fit(data)
    labels = kmeans.fit_predict(data)
    # process
    return labels


def do_clustering(data, method=None):
    if method is not None:
        config.set_clustering_method(method)
    clustering_method = select_method()
    labels = clustering_method(data)
    new_matrix = append_clustering_result(labels, distance_m=data)
    print(new_matrix)
    return new_matrix


def select_method(method=None):
    if method is None:
        method = config.cluster_method
    return {
        config.clusterMethodType.KMeans: KMeans_clustering,
        'default': KMeans_clustering,
    }.get(method, 'default')


# labels = KMeans_clustering(data)
def append_clustering_result(labels, distance_m):
    encoder = OneHotEncoder(categories='auto')
    onehot_labels = encoder.fit_transform(labels.reshape(-1, 1)).toarray()
    # Add the binary vector as a new feature to the original feature matrix
    new_clustering_matrix = np.hstack([distance_m, onehot_labels])
    # merge matrix here
    return new_clustering_matrix


if __name__ == '__main__':
    config.set_config(config.speciesType.human, config.chainType.alpha)
    config.set_distance_method(config.distanceMethodType.tcrdist)
    config.set_fe_method(config.feMethodType.distance_metrics)
    data = load_data('../../data/vdjdb_full.tsv').iloc[:200, :]
    distance_matrix = do_features_extraction(data)
    # print(distance_matrix)
    do_clustering(distance_matrix)
