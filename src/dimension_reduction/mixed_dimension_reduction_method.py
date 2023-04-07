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



def knn_umap_plot(feature,target):
    n_neighbors = 10
    random_state = 42
    X = feature
    y = target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state
    )

    # Reduce dimension to 2 with PCA
    pca = make_pipeline(StandardScaler(), PCA(n_components=2, random_state=random_state))

    # Reduce dimension to 2 with LinearDiscriminantAnalysis
    lda = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(n_components=2))

    # Reduce dimension to 2 with NeighborhoodComponentAnalysis
    nca = make_pipeline(
        StandardScaler(),
        NeighborhoodComponentsAnalysis(n_components=4, random_state=random_state),
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
            "PCA".format(name, n_neighbors, acc_knn)
        )
        plt.colorbar()

    plt.show()

    nca.fit(X_train, y_train)
    knn.fit(nca.transform(X_train), y_train)
    acc_knn = knn.score(model.transform(X_test), y_test)
    X_embedded = nca.transform(X)
    plt.title("NCA")
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, s=30, cmap="Set1")
    plt.colorbar()
    plt.show()


    m = "euclidean"

    umap_model = umap.UMAP(random_state=42, n_neighbors=20, min_dist=0.6, n_components=2, metric=m)
    embedding = umap_model.fit_transform(X_embedded)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=target, cmap='Spectral', s=10)
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar()
    plt.title('NCA+UMAP', fontsize=24)
    plt.show()

    '''
        fig = px.scatter(
            embedding[:, 0], embedding[:, 1],
            color=target.astype(str), labels={'color': 'target'}
        )
        fig.show()
    '''