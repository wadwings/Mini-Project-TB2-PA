import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from src.preprocess.setup import config
from src.distance_compute.distance_compute import *
from sklearn.metrics import silhouette_score





def check_best_k(data):
    k_range = range(2, 11)
    silhouette_scores = []
    for k in k_range:
        # 训练K-Means模型
        model = KMeans(n_clusters=k, random_state=42)
        model.fit(data)
        score = silhouette_score(data, model.labels_)
        silhouette_scores.append(score)
    # 找到最佳的K值
    best_k = k_range[silhouette_scores.index(max(silhouette_scores))]
    return best_k

    #
    # inertias = []
    # K = 10
    # for k in range(1, K + 1):
    #     kmm = KMeans(n_clusters=k).fit(distance_m)
    #     kmm.fit(distance_m)
    #     inertias.append(kmm.inertia_)
    # # Plot the elbow
    # plt.figure()
    # plt.plot(range(1, K + 1), inertias, 'bx-')
    # plt.xlabel('k')
    # plt.ylabel('Inertia')
    # plt.title('The elbow method showing the optimal k')


def KMeans_clustering(data):
    k = check_best_k(data)
    kmeans = KMeans(n_clusters=k, random_state=10).fit(data)
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_, cmap='rainbow')
    labels = kmeans.fit_predict(data)
    # process
    return labels



def do_clustering(data_, method=None):
    if method is not None:
        config.set_clustering_method(method)
    clustering_method = select_method()
    labels = clustering_method(data)
    new_matrix = append_clustering_result(distance_m=data)
    return labels, new_matrix


def select_method(method=None):
    if method is None:
        method = config.cluster_method
    return {
        config.clusterMethodType.KMeans: KMeans_clustering,
        'default': KMeans_clustering,
    }.get(method, 'default')


labels = KMeans_clustering(data)
def append_clustering_result(labels,distance_m):
    encoder = OneHotEncoder(categories='auto')
    onehot_labels = encoder.fit_transform(labels.reshape(-1, 1)).toarray()
    # Add the binary vector as a new feature to the original feature matrix
    new_clustering_matrix = np.hstack([distance_m, onehot_labels])
    # merge matrix here
    return new_clustering_matrix


if __name__ == '__main__':
    config.set_config(config.speciesType.human, config.chainType.alpha)
    data = load_data().iloc[:200, :]
    data = compute_count(data, config.get_columns())
    distance_matrix = compute_distance(data, data)
    D = distance_matrix
    dissimilarities = D.max() - D
    similarities = dissimilarities / dissimilarities.max()
    append_clustering_result(similarities)
