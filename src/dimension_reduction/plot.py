from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


def plot(data, labels, title, xlabel='X', ylabel='Y'):
    import matplotlib.pyplot as plt
    import numpy as np

    x = data[:, 0]
    y = data[:, 1]

    colors = {}

    n_labels = len(np.unique(labels))
    cmap = ListedColormap(plt.cm.get_cmap('tab20').colors)

    unique_labels = set(labels)
    index = 0
    for label in unique_labels:
        # generate a random RGB color
        color = cmap.colors[index]
        index = index + 1
        # map the label to the color
        colors[label] = color
    print(colors)

    # 绘制散点图
    fig, ax = plt.subplots()
    for label in colors:
        mask = np.array(labels) == label
        ax.scatter(x[mask], y[mask], c=colors[label], label=label)

    # 创建 color panel
    handles = [plt.scatter([], [], c=color, label=label) for label, color in colors.items()]
    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5), title='Labels')

    # 显示图形
    plt.show()


def fashion_scatters(x, colors):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x[:, 0], x[:, 1], c=colors, cmap='rainbow')
    return fig, ax
