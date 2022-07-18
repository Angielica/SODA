import matplotlib.pyplot as plt


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
