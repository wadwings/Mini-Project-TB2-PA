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
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA


def check_best_k(data):
    k_range = range(2, 11)
    silhouette_scores = []
    for k in k_range:
        # 训练K-Means模型
        t = data
        model = KMeans(n_clusters=k, random_state=42, n_init='auto')
        model.fit(t)
        score = silhouette_score(t, model.labels_,metric='cosine')
        silhouette_scores.append(score)

    # 找到最佳的K值
    print(f'scores: {silhouette_scores}')
    best_k = k_range[silhouette_scores.index(max(silhouette_scores))]
    print(f'Best k value:',best_k)
    print(f'The maximum silhouette coefficient：',max(silhouette_scores))
    return best_k

def check_best_k_h(data):
    k_range = range(2, 11)
    silhouette_scores = []
    print(data)
    for k in k_range:
        # 训练K-Means模型
        t = data
        model = AgglomerativeClustering(n_clusters=k, linkage='ward')
        model.fit(t)
        score = silhouette_score(t, model.labels_, metric='cosine')
        silhouette_scores.append(score)

    # 找到最佳的K值
    best_k_h = k_range[silhouette_scores.index(max(silhouette_scores))]
    print(f'H The maximum silhouette coefficient：',max(silhouette_scores))
    return best_k_h

def check_best_eps(data):
    # 定义一组eps值
    eps_values = np.linspace(0.1, 1.0, num=10)

    # 初始化最优参数
    best_eps = None
    best_score = -1
    for eps in eps_values:
        model = DBSCAN(eps=eps, min_samples=5)
        labels = model.fit_predict(data)
        print(labels)
    #     score = silhouette_score(data, labels,metric='cosine')
    #     if score > best_score:
    #         best_score = score
    #         best_eps = eps
    # print("Maximum silhouette score: {}".format(best_score))
    # return best_eps


def KMeans_clustering(data):
    k = check_best_k(data)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(data)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(data)
    labels = kmeans.fit_predict(data)
    # process
    return labels

def Hierarchical_clustering(data):
    k = check_best_k_h(data)
    model = AgglomerativeClustering(n_clusters=k, linkage='ward')
    model.fit(data)
    return model.labels_

def DBSCAN_clustering(data):
    e = check_best_eps(data)
    model = DBSCAN(eps=e, min_samples=5)
    labels = model.fit_predict(data)
    return labels

def PCA_Kmeans_improve_clustering(data):
    pca = PCA()
    pca.fit(data)
    variance_ratio = pca.explained_variance_ratio_
    cumulative_var_ratio = np.cumsum(variance_ratio)

    # 找到累计方差贡献率第一次超过70%的位置
    num_components = np.argmax(cumulative_var_ratio >= 0.7) + 1
    print(num_components)

    pca1 = PCA(n_components=num_components)
    pca_apply_kmeans = pca1.fit_transform(data)

    k = check_best_k(pca_apply_kmeans)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(pca_apply_kmeans)
    labels = kmeans.fit_predict(pca_apply_kmeans)
    print(f"kmeans labels", labels)
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
        config.clusterMethodType.Hierarchical: Hierarchical_clustering,
        config.clusterMethodType.DBSCAN: DBSCAN_clustering,
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
    data = load_data('../../data/vdjdb_full.tsv').iloc[:600, :]
    data, label = do_preprocess(data)
    print(data)
    distance_matrix = do_features_extraction(data)
    KMeans_clustering(distance_matrix)
    # clustering_matrix = do_clustering(distance_matrix)
    # print(f'clustering matrix: \n{clustering_matrix}')

