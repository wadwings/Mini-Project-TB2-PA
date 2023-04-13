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
        model = KMeans(n_clusters=k, random_state=42, n_init='auto')
        model.fit(t)
        score = silhouette_score(t, model.labels_)
        silhouette_scores.append(score)
    # 找到最佳的K值
    best_k = k_range[silhouette_scores.index(max(silhouette_scores))]
    return best_k


def KMeans_clustering(data):
    k = check_best_k(data)
    kmeans = KMeans(n_clusters=k, random_state=10, n_init='auto').fit(data)
    labels = kmeans.fit_predict(data)
    # process
    return labels


def do_clustering(data, method=None):
    if method is not None:
        config.set_clustering_method(method)
    clustering_method = select_method()
    labels = clustering_method(data)
    # print('labels:', labels)
    new_matrix = append_clustering_result(labels, distance_m=data)
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
    new_clustering_matrix = np.append(distance_m, np.reshape(labels, (-1, 1)), axis=1)
    # merge matrix here
    return new_clustering_matrix


if __name__ == '__main__':
    config.set_config(config.speciesType.human, config.chainType.alpha)
    config.set_distance_method(config.distanceMethodType.tcrdist)
    config.set_fe_method(config.feMethodType.distance_metrics)
    data = load_data('../../data/vdjdb_full.tsv').iloc[:200, :]
    data, label = do_preprocess(data)
    distance_matrix = do_features_extraction(data)
    clustering_matrix = do_clustering(distance_matrix)
    print(f'clustering matrix: \n{clustering_matrix}')

