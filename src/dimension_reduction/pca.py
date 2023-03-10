from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# def pca_decomposition(data):
#     clf = PCA(n_components=2)
#     x = clf.fit_transform(data)


def pca_plot(feature, target):

    pca = PCA(n_components=2)

    X_pca = pca.fit_transform(feature)
    label = target.values
    # plt.figure(4)

    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=label, cmap='viridis')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar()

    plt.title('Projected data showing target')
    plt.show()
