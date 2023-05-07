from sklearn.decomposition import PCA
import numpy as np
from sklearn.cluster import KMeans
from src.dimension_reduction.method import pca_method,tsne_method
from src.distance_compute.distance_compute import *
from src.features_extraction.features_extraction import do_features_extraction
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
import numpy as np
from sklearn import metrics

# def check_best_pca_n(data):
#     pca = PCA()
#     pca.fit(data)
#     variance_ratio = pca.explained_variance_ratio_
#     cumulative_var_ratio = np.cumsum(variance_ratio)
#
#     # 找到累计方差贡献率第一次超过70%的位置
#     num_components = np.argmax(cumulative_var_ratio >= 0.7) + 1
#     print(num_components)
#     return num_components

# Find best k for keams
# def check_best_k(data):
#     k_range = range(2, 11)
#     silhouette_scores = []
#     for k in k_range:
#         t = data
#         model = KMeans(n_clusters=k, random_state=42, n_init='auto')
#         model.fit(t)
#         score = silhouette_score(t, model.labels_,metric='cosine')
#         silhouette_scores.append(score)
#
#     print(f'scores: {silhouette_scores}')
#     best_k = k_range[silhouette_scores.index(max(silhouette_scores))]
#     print(f'Best k value:',best_k)
#     print(f'The maximum silhouette coefficient：',max(silhouette_scores))
#     return best_k

# Find best k for Hierarchical
# def check_best_k_h(data):
#     k_range = range(2, 11)
#     silhouette_scores = []
#     print(data)
#     for k in k_range:
#         t = data
#         model = AgglomerativeClustering(n_clusters=k, linkage='ward')
#         model.fit(t)
#         score = silhouette_score(t, model.labels_, metric='cosine')
#         silhouette_scores.append(score)
#
#     best_k_h = k_range[silhouette_scores.index(max(silhouette_scores))]
#     print(f'H The maximum silhouette coefficient：',max(silhouette_scores))
#     return best_k_h


def purity_score(ytrue, ypred):
    from sklearn.metrics.cluster import contingency_matrix
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(ytrue, ypred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)



# 4 Clustering Method
def KMeans_clustering(data,truelabel):
    kmeans = KMeans(n_clusters=10, random_state=42, n_init='auto').fit(feature_matrix)
    labels = kmeans.fit_predict(feature_matrix)
    purity_kmeans = purity_score(truelabel, labels)
    print("purity kMeans:",purity_kmeans)
    # process
    return labels

def Hierarchical_clustering(data,truelabel):
    # k = check_best_k_h(data)
    model = AgglomerativeClustering(n_clusters=10, linkage='average')
    labels = model.fit_predict(feature_matrix)
    purity_hierarchical = purity_score(truelabel, labels)
    print("purity Hierarchical:", purity_hierarchical)
    return labels



def PCA_Kmeans_improve_clustering(data):
    pca_apply_kmeans = pca_method(data)
    labels = KMeans_clustering(pca_apply_kmeans)
    return pca_apply_kmeans, labels

def tsne_Kmeans_improve_clustering(data):
    tsne_apply_kmeans = tsne_method(data)
    labels = KMeans_clustering(tsne_apply_kmeans)
    return tsne_apply_kmeans, labels


# def append_clustering_result(labels, distance_m):
#     new_clustering_matrix = np.append(distance_m, np.reshape(labels, (-1, 1)), axis=1)
#     # merge matrix here
#     return new_clustering_matrix

# def do_clustering(data, method=None):
#     if method is not None:
#         config.set_clustering_method(method)
#     clustering_method = select_method()
#     labels = clustering_method(data)
#     # print('labels:', labels)
#     # new_matrix = append_clustering_result(labels, distance_m=data)
#     # return new_matrix

def linked_plot(data):
    # linked = linkage(feature_matrix, 'single')
    # labelList = range(len(feature_matrix))
    # plt.figure()
    # dendrogram(linked, labels=labelList)
    # plt.show()
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    metrics = ['single', 'average', 'complete', 'centroid']
    for i in range(2):
        for j in range(2):
            linked = linkage(data, metrics[(i * 2) + j])
            dendrogram(linked, ax=axs[i, j])
            axs[i, j].set_title(metrics[(i * 2) + j])



# Two plot method
def basic_clustering_plot(data,method=None):
    if method is not None:
        config.set_clustering_method(method)
    clustering_method = select_basic_method()
    print(clustering_method)
    labels = clustering_method(data)
    data1 = pd.DataFrame(data)
    # process
    clusters_scale = pd.concat([data1, pd.DataFrame({'cluster_scaled':labels})], axis=1)
    pca2 = PCA(n_components=2).fit(data1)
    pca2d = pca2.transform(data1)
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x = pca2d[:, 0], y =pca2d[:, 1],
                    hue=labels,
                    palette='Set1',
                    s=100, alpha=0.2,
                    ).set_title('KMeans Clusters Derived from Original Dataset', fontsize=15)
    plt.legend()
    plt.ylabel('PC2')
    plt.xlabel('PC1')
    plt.show()



def KMeans_improve_plot(data,method=None):
    if method is not None:
        config.set_clustering_method(method)
    clustering_method = select_improve_method()
    print(clustering_method)
    pca_scale, labels_pca_scale = clustering_method(data)
    pca_df_scale = pd.DataFrame(pca_scale)
    clusters_pca_scale = pd.concat([pca_df_scale, pd.DataFrame({'pca_clusters': labels_pca_scale})], axis=1)
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x=clusters_pca_scale.iloc[:, 0], y=clusters_pca_scale.iloc[:, 1], hue=labels_pca_scale, palette='Set1',
                        s=100, alpha=0.2).set_title('KMeans Clusters Derived from PCA', fontsize=15)
    plt.legend()
    plt.show()


def select_basic_method(method=None):
    if method is None:
        method = config.cluster_method
    return {
        config.clusterMethodType.KMeans: KMeans_clustering,
        config.clusterMethodType.Hierarchical: Hierarchical_clustering,
        # config.clusterMethodType.DBSCAN: DBSCAN_clustering,
        'default': KMeans_clustering,
    }.get(method, 'default')


def select_improve_method(method=None):
    if method is None:
        method = config.cluster_method
    return {
        config.clusterMethodType.pca_add_KMeans: PCA_Kmeans_improve_clustering,
        config.clusterMethodType.tsne_add_KMeans: tsne_Kmeans_improve_clustering,
        'default': PCA_Kmeans_improve_clustering,
    }.get(method, 'default')





if __name__ == '__main__':
    config.set_config(config.speciesType.human, config.chainType.beta)
    config.set_label(config.labelType.epitope)
    config.set_fe_method(config.feMethodType.distance_metrics)
    config.set_distance_method(config.distanceMethodType.tcrdist)

    data = load_data("../../data/vdjdb_full.tsv")
    data, label = do_preprocess(data)
    data = do_segmentation(data, 10)
    feature_matrix = do_features_extraction(data)

    KMeans_clustering(feature_matrix,data['label'])
    Hierarchical_clustering(feature_matrix,data['label'])
    linked_plot(feature_matrix)

    # print("Kmeans==========================================================================>")
    # basic_clustering_plot(feature_matrix, method='kmeans')
    # print("PCA + Kmeans==========================================================================>")
    # KMeans_improve_plot(feature_matrix, method='pca_add_KMeans')




