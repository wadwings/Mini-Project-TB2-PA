from sklearn.decomposition import PCA, KernelPCA, SparsePCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS
from sklearn.decomposition import PCA,KernelPCA,SparsePCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import numpy as np
from sklearn.preprocessing import QuantileTransformer
import umap
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler



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
    tsne = TSNE(n_components=2, metric='precomputed',init='random', n_iter=500000, learning_rate=1e-7)
    embedding = tsne.fit_transform(features)
    return embedding


def umap_method(features):
    m = "euclidean"
    umap_model = umap.UMAP(random_state=42, n_neighbors=10, min_dist=0.2, n_components=2, metric=m)
    embedding = umap_model.fit_transform(features)
    return embedding

def knn_umap_method(feature,target):
    n_neighbors = 10
    random_state = 42
    X = feature
    y = target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state
    )

    nca = make_pipeline(
        StandardScaler(),
        NeighborhoodComponentsAnalysis(n_components=4, random_state=random_state),
    )
    m = "euclidean"
    # Use a nearest neighbor classifier to evaluate the methods
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    nca.fit(X_train, y_train)
    knn.fit(nca.transform(X_train), y_train)
    acc_knn = knn.score(nca.transform(X_test), y_test)
    X_embedded = nca.transform(X)
    umap_model = umap.UMAP(random_state=42, n_neighbors=20, min_dist=0.6, n_components=2, metric=m)
    embedding = umap_model.fit_transform(X_embedded)
    return embedding



def mds_method(features):
    mds = MDS(n_components=2, random_state=42)
    embedding = mds.fit_transform(features)
    return embedding
