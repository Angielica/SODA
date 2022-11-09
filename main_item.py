import sys

import torch
import torch.nn as nn
import matplotlib
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

from models.sparse_ae import SparseAutoEncoder
from models.trainer_item import Trainer
from utils.utility import get_device, make_dir, testing, create_color_map, data_mapping
from utils.plotter import visualize_tsne
from utils.data_generator import get_data

from sklearn.manifold import TSNE
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
import warnings
import random
warnings.filterwarnings("ignore")
matplotlib.style.use('ggplot')

    
def main(f_name):
    
    with open(f_name, "r") as fp:
        params = json.load(fp)

    seed = 230788

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms = True
    torch.backends.cudnn.benchmark = False

    device = get_device()
    make_dir()
    params['device'] = device
    
    h_dim = params['h_dim']
    dataset_name = params['dataset']
    is_disc = params['is_disc']
    rec_disc = params['rec_disc']
    epochs = params['epochs']
    
    print(f'Dataset: {dataset_name}')
    train_loader, test_loader, test_original, params = get_data(params)
    
    n_items = params['n_items']
    z_dim = int(n_items * 1.5)  # 1.5
    params['z_dim'] = z_dim

    model = SparseAutoEncoder(x_dim=n_items, h_dim=h_dim, z_dim=z_dim, is_disc=is_disc,
                              rec_disc=rec_disc).to(device)
    # the loss function
    criterion = nn.MSELoss()
    path_model = f"outputs/checkpoint/SDAE_{dataset_name}_z_dim_higher.pth"

    print('Training ...')
    trainer = Trainer(model, criterion, params)
    trainer.train(train_loader, test_loader, epochs, path_model)

    # Load model
    print('Testing ...')
    model.load_state_dict(torch.load(path_model))

    patterns = pd.read_csv(params['ifm'], header=None)

    association = dict()
    test = test_original
    test['y'] = 0
    model.eval()
    lab = []
    targets = []
    for idx in range(patterns.shape[0]):
        patt = str(patterns.loc[idx]).split()[1:-4]

        x = test.drop(columns=['y'])

        for elem in patt:
            x = x[x[elem] == 1]

        test.loc[x.index, 'y'] = idx+1
        lab.append(idx+1)
        targets.append(str(idx+1))

        x = torch.tensor(x.values, dtype=torch.float).to(device)

        with torch.no_grad():
            _, _, z = model(x) # per comodità z_hard l'ho chiamata z

        association[str(patt)] = {'x': x, 'z': z}

    # Per ogni pattern, z associata, se due pattern stessa z, controllare x: se x uguale Ok, altrimenti No
    # Verificare se le x hanno più pattern

    x_test = test.drop(columns=['y']).values
    y_test = test['y'].values

    x_test = torch.tensor(x_test, dtype=torch.float)
    y_test = torch.tensor(y_test, dtype=torch.float)

    test_loader = DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=params['bs'])

    encoding, labels = testing(model, test_loader, params)
    path = f'outputs/images/tsne_{dataset_name}_z_dim_higher.png'

    def get_cmap(n, name='hsv'):
        '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
        RGB color; the keyword argument name must be a standard mpl colormap name.'''
        return plt.cm.get_cmap(name, n)

    col = get_cmap(len(np.unique(labels)))

    visualize_tsne_2(encoding, labels, col, lab, targets, path)
    visualizePCA_2(encoding, labels, col, lab, targets, f'outputs/images/pca_{dataset_name}_z_dim_higher.png')
    print()
    
    
def visualize_tsne_2(embeddings, classes, colors, label, targets, path=None):
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

    for l in label:
        indicesToKeep = tsne_df['label'] == l
        ax.scatter(tsne_df.loc[indicesToKeep, 'tsne-2d-one'],
                   tsne_df.loc[indicesToKeep, 'tsne-2d-two'],
                   c=colors(l), s=20)

    ax.legend(targets)
    plt.savefig(path)


def visualizePCA_2(embeddings, classes, colors, label, targets, path=None):
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

    for l in label:
        indicesToKeep = pca_df['label'] == l
        ax.scatter(pca_df.loc[indicesToKeep, 'pca-one'],
                   pca_df.loc[indicesToKeep, 'pca-two'],
                   c=colors(l), s=8, cmap='tab10')

    # for l in label:
    #     indicesToKeep = pca_df['label'] == l
    #    ax.scatter(pca_df.loc[indicesToKeep, 'pca-one'],
    #                pca_df.loc[indicesToKeep, 'pca-two'],
    #                c=colors(l))

    ax.legend(targets)
    plt.savefig(path)

    
if __name__ == '__main__':
    main(sys.argv[1])
