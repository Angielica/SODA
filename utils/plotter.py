import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from matplotlib.patches import Rectangle

import torch


def my_heat_map(itemsets, unique_colors, model, device, params, special_point=None, n_dots=5000,
                min_alpha=.1, alpha_coeff=1, x_min=0, x_max=1, y_min=0, y_max=1):

    n_itemsets = len(itemsets)
    unique_colors = np.array(unique_colors)
    itemsets = [torch.tensor(x, device=device) for x in itemsets]
    noise = torch.rand(n_dots, params['z_dim'], device=device)

    noise[:, :] = (x_max - x_min) * noise[:, :] + x_min
    #noise[:, 1] = (y_max - y_min) * noise[:, 1] + y_min
    recon = model.generate(noise)

    diff = torch.zeros(n_dots, n_itemsets, device=device)

    for j in range(n_itemsets):
        diff[:, j] = torch.sum(torch.abs(recon - itemsets[j]), axis=1)

    diff = diff.detach().cpu().numpy() # --> Aggiunto da my_heat_map di GAN_SIGMOID
    noise = noise.detach().cpu().numpy()
    class_index = diff.argmin(axis=1)  #.detach().cpu().numpy()

    alphas = diff[range(n_dots), class_index]
    min_val = alphas.min()
    max_val = alphas.max()
    if min_val == max_val:
        alphas.fill(1.)
    else:
        alphas = (alphas - min_val) / (alphas.max() - min_val)
        alphas = np.exp(-alpha_coeff * alphas)
        alphas = alphas * (1 - min_alpha) + min_alpha
    #alphas = alphas.detach().cpu().numpy()

    colors_to_plot = np.array([colors.to_rgba(x) for x in unique_colors[class_index]])
    colors_to_plot[:, 3] = alphas

    plt.figure(figsize=(10, 10))

    plt.scatter(noise[:, 0], noise[:, 1], s=5, color=colors_to_plot, marker='o')

    if special_point is not None:
        print(special_point)
        special_point = special_point.detach().cpu().numpy()
        plt.scatter(special_point[:, 0], special_point[:, 1], s=200, color='r', marker='*')

    plt.grid(True)
    plt.title('Embedding plot')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # plt.savefig(os.path.join(cwd, 'data/vae_data.pdf'), bbox_inches='tight')
    plt.show()


def plot_colortable(color_map, title, n_items, sort_colors=True, emptycols=0):
    cell_width = 212
    cell_height = 22
    swatch_width = 48
    margin = 12
    topmargin = 40

    # Sort colors by hue, saturation, value and name.
    if sort_colors is True:
        by_hsv = sorted((tuple(colors.rgb_to_hsv(colors.to_rgb(color))), name)
                        for name, color in color_map.items())
        names = [name for hsv, name in by_hsv]
    else:
        names = list(color_map)

    n = len(names)
    ncols = 4 - emptycols
    nrows = n // ncols + int(n % ncols > 0)

    width = cell_width * 4 + 2 * margin
    height = cell_height * nrows + margin + topmargin
    dpi = 72

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(margin/width, margin/height,
                        (width-margin)/width, (height-topmargin)/height)
    ax.set_xlim(0, cell_width * 4)
    ax.set_ylim(cell_height * (nrows-0.5), -cell_height/2.)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()
    ax.set_title(title, fontsize=24, loc="left", pad=10)

    for i, name in enumerate(names):
        row = i % nrows
        col = i // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col
        text_pos_x = cell_width * col + swatch_width + 7

        tmp = [int(x) for x in bin(name)[2:]]
        text_name = np.pad(tmp, (n_items - len(tmp), 0),
                           'constant', constant_values=(0, 0))

        ax.text(text_pos_x, y, text_name, fontsize=14,
                horizontalalignment='left',
                verticalalignment='center')

        ax.add_patch(
            Rectangle(xy=(swatch_start_x, y-9), width=swatch_width,
                      height=18, facecolor=color_map[name], edgecolor='0.7'))

    return fig


def plot_loss(train_loss, val_loss, path):
    plt.plot(train_loss, label='train')
    plt.plot(val_loss, label='val')
    plt.legend()
    plt.savefig(path)


def visualize_tsne(embeddings, classes, colors, label, targets, path=None):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import pandas as pd

    tsne = TSNE(n_components=2, random_state=42)
    print('Fitting ...')
    tsne_emb = tsne.fit_transform(embeddings)

    print('Visualization')
    tsne_df = pd.DataFrame()
    tsne_df['tsne-2d-one'] = tsne_emb[:, 0]
    tsne_df['tsne-2d-two'] = tsne_emb[:, 1]
    tsne_df['label'] = classes

    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(1, 1, 1)

    ax.set_xlabel('tsne-2d-one')
    ax.set_ylabel('tsne-2d-two')

    for l, c in zip(label, colors):
        indicesToKeep = tsne_df['label'] == l
        ax.scatter(tsne_df.loc[indicesToKeep, 'tsne-2d-one'],
                   tsne_df.loc[indicesToKeep, 'tsne-2d-two'],
                   c=c, s=20)

    ax.legend(targets)
    plt.savefig(path)


def visualizePCA(embeddings, classes, colors, label, targets, path=None):
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    pca = PCA(n_components=2, random_state=113)
    pca_res = pca.fit_transform(embeddings)

    pca_df = pd.DataFrame()
    pca_df['pca-one'] = pca_res[:, 0]
    pca_df['pca-two'] = pca_res[:, 1]
    pca_df['label'] = classes

    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(1, 1, 1)

    ax.set_xlabel('pca-one')
    ax.set_ylabel('pca-two')

    for l, c in zip(label, colors):
        indicesToKeep = pca_df['label'] == l
        ax.scatter(pca_df.loc[indicesToKeep, 'pca-one'],
                   pca_df.loc[indicesToKeep, 'pca-two'],
                   c=c)

    ax.legend(targets)
    plt.savefig(path)

