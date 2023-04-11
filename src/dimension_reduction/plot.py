from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
import colorsys


def plot(data, labels, title, xlabel='X', ylabel='Y'):
    import matplotlib.pyplot as plt
    import numpy as np

    x = data[:, 0]
    y = data[:, 1]

    colors = {}

    n_labels = len(np.unique(labels))
    cmap = ListedColormap(plt.cm.get_cmap('tab20').colors)

    # 将cmap转换为列表
    colors = cmap.colors

    # 将颜色值转换为RGB值
    rgb_colors = [mcolors.to_rgb(color) for color in colors]

    # 循环遍历RGB颜色并提取HLS值
    hls_colors = [colorsys.rgb_to_hls(*rgb_color) for rgb_color in rgb_colors]

    # 调整H值以获得更多的颜色
    new_hls_colors = []
    for hls_color in hls_colors:
        h, l, s = hls_color
        for t in range(1, 10):
            new_h = (h + t/10) % 1.0  # 调整H值以获得新颜色
            new_hls_color = (new_h, l, s)
            new_hls_colors.append(new_hls_color)

    # 将HLS值转换回RGB值
    new_rgb_colors = [colorsys.hls_to_rgb(*new_hls_color) for new_hls_color in new_hls_colors]

    # 将RGB值转换为标准颜色编码
    new_colors = [mcolors.to_hex(new_rgb_color) for new_rgb_color in new_rgb_colors]

    print(new_colors)
    colors = {}
    unique_labels = set(labels)
    index = 0
    for label in unique_labels:
        # generate a random RGB color
        color = new_colors[index]
        index = index + 1
        # map the label to the color
        colors[label] = color
    print(colors)

    print(f"x: \n{x}")
    print(f"y: \n{y}")
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
