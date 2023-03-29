from sklearn.decomposition import PCA,KernelPCA,SparsePCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from sklearn.manifold import MDS

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
    feature = switch_standard_data(feature)

    pca = PCA(n_components=2,random_state=42)
    X_pca = pca.fit_transform(feature)
    label = target.values
    # plt.figure(4)

    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=label, cmap='viridis')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar()

    plt.title('PCA')
    plt.show()
def pca_plot1(feature,target):
    feature = switch_standard_data(feature)

    pca = PCA(n_components=2,random_state=42)
    X_pca = pca.fit_transform(feature)
    label = target
    # plt.figure(4)

    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=label, cmap='viridis')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar()

    plt.title('Force Assignment PCA (Target: Antigen Species)')
    plt.show()

def kernel_pca(feature,target):
    kernel_pca = KernelPCA(n_components=7, kernel='linear')
    X_kernel_pca = kernel_pca.fit_transform(feature)

    label = target
    # plt.figure(4)

    plt.scatter(X_kernel_pca[:, 0], X_kernel_pca[:, 1], c=label, cmap='viridis')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar()

    plt.title('Kernel PCA')
    plt.show()



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
    tsne = TSNE(n_components=2, metric='precomputed')
    X_tsne = tsne.fit_transform(feature)

    fig, ax = fashion_scatter(X_tsne, colors)
    plt.title('Tsne')
    plt.show()


    '''
    fig = px.scatter(
        X_tsne,
        color=colors, labels={'color': 'mhc class'}
    )
    fig.show()

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

    


def umap_plot(feature, target):

    m = "euclidean"

    umap_model = umap.UMAP(random_state=42, n_neighbors=10, min_dist=0.2, n_components=2, metric=m)
    embedding = umap_model.fit_transform(feature)

    plt.scatter(embedding[:, 0], embedding[:, 1], c=target, cmap='Spectral', s=5)
    plt.colorbar()
    plt.title('Force Assignment UMAP (Target: Antigen Species)')
    plt.show()


def mds_plot(feature,target):
    mds = MDS(n_components=2,random_state=42)
    X = mds.fit_transform(feature)
    label = target

    plt.scatter(X[:, 0], X[:, 1], c=label, cmap='viridis')
    plt.title('MDS')
    plt.show()


