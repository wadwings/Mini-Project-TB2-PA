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





import pandas as pd
import numpy as np


# def pca_decomposition(data):
#     clf = PCA(n_components=2)
#     x = clf.fit_transform(data)

switch_standard = 0
def switch_standard_data(feature):

    if switch_standard  == 0:
        return feature
    else:
        scaler = StandardScaler()
        feature = scaler.fit_transform(feature)
        return feature


def pca_plot(feature,target):
    n_neighbors = 40
    random_state = 42
    X = feature
    y = target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=random_state
    )

    # Reduce dimension to 2 with PCA
    pca = make_pipeline(StandardScaler(), PCA(n_components=2, random_state=random_state))

    # Reduce dimension to 2 with LinearDiscriminantAnalysis
    lda = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(n_components=2))

    # Reduce dimension to 2 with NeighborhoodComponentAnalysis
    nca = make_pipeline(
        StandardScaler(),
        NeighborhoodComponentsAnalysis(n_components=3, random_state=random_state),
    )

    # Use a nearest neighbor classifier to evaluate the methods
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Make a list of the methods to be compared
    dim_reduction_methods = [("PCA", pca), ("LDA", lda), ("NCA", nca)]

    # plt.figure()
    for i, (name, model) in enumerate(dim_reduction_methods):
        plt.figure()
        # plt.subplot(1, 3, i + 1, aspect=1)

        # Fit the method's model
        model.fit(X_train, y_train)

        # Fit a nearest neighbor classifier on the embedded training set
        knn.fit(model.transform(X_train), y_train)

        # Compute the nearest neighbor accuracy on the embedded test set
        acc_knn = knn.score(model.transform(X_test), y_test)

        # Embed the data set in 2 dimensions using the fitted model
        X_embedded = model.transform(X)

        # Plot the projected points and show the evaluation score
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, s=30, cmap="Set1")
        plt.title(
            "{}, KNN (k={})\nTest accuracy = {:.2f}".format(name, n_neighbors, acc_knn)
        )
    plt.show()

    nca.fit(X_train, y_train)
    knn.fit(nca.transform(X_train), y_train)
    acc_knn = knn.score(model.transform(X_test), y_test)
    X_embedded = nca.transform(X)
    plt.title(
        "TEST UMAP{}, KNN (k={})\nTest accuracy = {:.2f}".format(name, n_neighbors, acc_knn)
    )
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, s=30, cmap="Set1")
    plt.show()


    m = "euclidean"

    umap_model = umap.UMAP(random_state=42, n_neighbors=20, min_dist=0.6, n_components=2, metric=m)
    embedding = umap_model.fit_transform(X_embedded)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=target, cmap='Spectral', s=5)
    plt.title('UMAP')
    plt.show()

'''
def pca_plot(feature,target):
    feature = switch_standard_data(feature)

    pca = PCA(n_components=2, svd_solver="randomized", n_oversamples= 50,power_iteration_normalizer='LU',random_state=42)
    X_pca = pca.fit_transform(feature)
    label = target.values
    # plt.figure(4)

    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=label, cmap='viridis')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar()

    plt.title('PCA')
    plt.show()
'''

'''
def SparsePCA_plot(feature,target):
    feature = switch_standard_data(feature)
    
    nca = make_pipeline(
        StandardScaler(),
        NeighborhoodComponentsAnalysis(n_components=2, random_state=random_state),
    )
    X_nca = pca.fit_transform(feature)
    label = target.values
    
    # plt.figure(4)

    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=label, cmap='viridis')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar()

    plt.title('PCA')
    plt.show()
'''

def tsne_plot(feature, target):
    def fashion_scatter(x, colors):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(x[:, 0], x[:, 1], c=colors, cmap='rainbow')
        return fig, ax

    colors = target
    RS = 42
    tsne = TSNE(n_components=2, perplexity=3)
    X_tsne = tsne.fit_transform(feature)

    fig, ax = fashion_scatter(X_tsne, colors)
    plt.show()
'''
    def fashion_scatter(x, colors):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(x[:, 0], x[:, 1], c=colors, cmap='rainbow')
        return fig, ax

    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(feature)
    tsne = TSNE(random_state=42, perplexity=3)
    X_tsne = tsne.fit_transform(X_pca)
    colors = target

    fig, ax = fashion_scatter(X_tsne, colors)
    plt.show()
'''

    

'''
def umap_plot(feature, target):

    m = "euclidean"

    umap_model = umap.UMAP(random_state=42, n_neighbors=100, min_dist=0.1, n_components=2, metric=m)
    embedding = umap_model.fit_transform(feature)

    plt.scatter(embedding[:, 0], embedding[:, 1], c=target, cmap='Spectral', s=5)
    plt.title('UMAP')
    plt.show()

'''

