from sklearn.decomposition import PCA, KernelPCA, SparsePCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS


# def pca_decomposition(data):
#     clf = PCA(n_components=2)
#     x = clf.fit_transform(data)

def pca_method(features):
    pca = PCA(n_components=2, random_state=42)
    embedding = pca.fit_transform(features)
    return embedding


def kpca_method(features):
    kernel_pca = KernelPCA(n_components=2, kernel='linear')
    embedding = kernel_pca.fit_transform(features)
    return embedding


def tsne_method(features):
    tsne = TSNE(n_components=2, metric='precomputed')
    embedding = tsne.fit_transform(features)
    return embedding


def umap_method(features):
    m = "euclidean"
    umap_model = umap.UMAP(random_state=42, n_neighbors=10, min_dist=0.2, n_components=2, metric=m)
    embedding = umap_model.fit_transform(features)
    return embedding


def mds_method(features):
    mds = MDS(n_components=2, random_state=42)
    embedding = mds.fit_transform(features)
    return embedding
