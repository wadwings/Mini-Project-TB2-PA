from matplotlib import pyplot as plt


def plot(data, labels, title, xlabel='X', ylabel='Y'):
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='Spectral', s=5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar()
    plt.title(title)
    plt.show()


def fashion_scatters(x, colors):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x[:, 0], x[:, 1], c=colors, cmap='rainbow')
    return fig, ax
