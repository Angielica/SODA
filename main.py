import sys

import torch
import torch.nn as nn
import matplotlib
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

from models.sparse_ae import SparseAutoEncoder
from models.trainer import Trainer
from utils.utility import get_device, make_dir, testing, create_color_map, data_mapping
from utils.plotter import visualize_tsne, plot_colortable, my_heat_map
from utils.data_generator import get_mnist, generate10patterns, get_loader, generate_10Patterns_MoreItems, \
    generate2patterns

from sklearn.manifold import TSNE
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import json

import pandas as pd

import warnings
warnings.filterwarnings("ignore")

matplotlib.style.use('ggplot')


def attempt(model, params, dataset, test_loader):
    device = params['device']
    color_map, color_list = createColorMap(params, dataset)
    print('##########')
    color_plot = plot_colortable(color_map, "Legend",
                                 n_items=params['n_items'])
    print('##########')

    two_d_data, reconstruction = data_mapping(model, test_loader, params)

    clusters = plt.figure(figsize=(12, 6))
    plt.scatter(two_d_data[:, 0], y=two_d_data[:, 1], s=5,
                color=color_list, marker="o")
    plt.grid(True)
    plt.title('Embedding plot')
    plt.show()

    # plt.show(color_plot)

    managed_fig = plt.figure()
    canvas_manager = managed_fig.canvas.manager
    canvas_manager.canvas.figure = color_plot
    color_plot.set_canvas(canvas_manager.canvas)
    plt.show()

    print()
    print(dataset['c_test_df'])

    c_rec_df = pd.DataFrame(reconstruction,
                            columns=dataset['columns'])
    c_rec_df = c_rec_df.groupby(dataset['columns']).size().reset_index(name='count')
    print(c_rec_df.sort_values(by=dataset['columns']))
    print(dataset['c_test_df'].sort_values(dataset['columns']))

    model.eval()
    with torch.no_grad():
        outlier1 = torch.tensor([1., 0., 0., 1.])
        outlier1 = outlier1.to(device)
        z1, _ = model(outlier1)

    z1 = z1.detach().cpu().numpy()

    clusters = plt.figure(figsize=(10, 10))
    plt.scatter(two_d_data[:, 0], y=two_d_data[:, 1], s=5,
                color=color_list, marker="o")
    plt.scatter(z1[0], z1[1], s=200, color='r', marker='*')
    plt.grid(True)
    plt.title('Embedding plot')
    plt.show()

    # plt.show(color_plot)

    managed_fig = plt.figure()
    canvas_manager = managed_fig.canvas.manager
    canvas_manager.canvas.figure = color_plot
    color_plot.set_canvas(canvas_manager.canvas)
    plt.show()

    torch.round(model.generate(torch.tensor([-4.5] * params['z_dim'], device=device)))

    itemsets = []
    domain_colors = []

    for item_id in dataset['chosen_itemsets']:
        tmp = [int(x) for x in bin(item_id)[2:]]
        tmp = np.pad(tmp, (params['n_items'] - len(tmp), 0), 'constant',
                     constant_values=(0, 0))
        itemsets.append(tmp)
        domain_colors.append(color_map[item_id])

    z1 = None

    my_heat_map(itemsets, domain_colors, model, device, params, alpha_coeff=10,
                x_min=-6, x_max=1, y_min=-6, y_max=1,
                n_dots=20000, special_point=z1)

    managed_fig = plt.figure()
    canvas_manager = managed_fig.canvas.manager
    canvas_manager.canvas.figure = color_plot
    color_plot.set_canvas(canvas_manager.canvas)
    plt.show()

    my_heat_map(itemsets, domain_colors, model, device, params, alpha_coeff=10,
                x_min=-10, x_max=10, y_min=-10,
                y_max=15, n_dots=20000)

    
def main(f_name):
    
    with open(f_name, "r") as fp:
        params = json.load(fp)

    device = get_device()
    make_dir()
    params['device'] = device
    
    n_items = params['n_items']
    h_dim = params['h_dim']
    dataset_name = params['dataset']
    is_disc = params['is_disc']
    rec_disc = params['rec_disc']
    epochs = params['epochs']
    params['z_dim'] = int(n_items * 1.5)

    model = SparseAutoEncoder(x_dim=n_items, h_dim=h_dim, z_dim=int(n_items * 1.5), is_disc=is_disc,
                              rec_disc=rec_disc).to(device)
    # the loss function
    criterion = nn.MSELoss()
    path_model = f"outputs/checkpoint/SDAE_{dataset_name}.pth"
    
    print(f'Dataset: {dataset_name}')
    train_loader, test_loader, dataset = None, None, None
    col, lab, targets = None, None, None
    
    if dataset_name == 'MNIST':
        train_loader, test_loader = get_mnist()
        col = ['coral', 'limegreen', 'royalblue', 'slateblue', 'orchid', 'palevioletred', 'chocolate', 'olive',
               'palegreen', 'teal']
        lab = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        targets = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    elif dataset_name == '10Patterns':
        dataset = generate10patterns(params)
        train_loader, val_loader, test_loader = get_loader(dataset, params)
        col = ['coral', 'limegreen', 'royalblue', 'slateblue', 'orchid', 'palevioletred', 'chocolate', 'olive',
               'palegreen', 'teal']
        lab = dataset['chosen_itemsets'] 
        targets = ['11', '5', '14', '8', '12', '1', '6', '13', '4', '10']

    elif dataset_name == '10PatternsMoreItems':
        dataset = generate_10Patterns_MoreItems(params)
        train_loader, val_loader, test_loader = get_loader(dataset, params)
        col = ['coral', 'limegreen', 'royalblue', 'slateblue', 'orchid', 'palevioletred', 'chocolate', 'olive',
               'palegreen', 'teal']
        lab = dataset['chosen_itemsets']
        targets = ['11', '5', '14', '8', '12', '1', '6', '13', '4', '10']
    elif dataset_name == '2Patterns':
        dataset = generate2patterns(params)
        train_loader, val_loader, test_loader = get_loader(dataset, params)
        col = ['red', 'blue']
        lab = dataset['chosen_itemsets']
        targets = ['3', '12']

    print('Training ...')
    trainer = Trainer(model, criterion, params)
    trainer.train(train_loader, test_loader, epochs, path_model)

    # Load model
    print('Testing ...')
    model.load_state_dict(torch.load(path_model))
    encoding, labels = testing(model, test_loader, params)
    path = f'outputs/images/tsne_{dataset_name}.png'

    visualize_tsne(encoding, labels, col, lab, targets, path)

    # if dataset_name == '10Patterns':
    #     attempt(model, params, dataset, test_loader)
    
    
if __name__ == '__main__':
    main(sys.argv[1])
