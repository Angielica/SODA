import torch
import torch.nn as nn
import matplotlib
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

from models.sparse_ae import SparseAutoEncoder
from models.trainer import Trainer
from utils.utility import get_device, make_dir
from utils.plotter import visualize_tsne

from sklearn.manifold import TSNE
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

import warnings
warnings.filterwarnings("ignore")

matplotlib.style.use('ggplot')


if __name__ == '__main__':
    epochs = 50
    reg_param = 0.001
    rho = .3
    add_sparsity = 'yes'
    num_enc_layer = 3
    num_dec_layer = 3
    learning_rate = 0.001
    batch_size = 100
    is_kl = False
    is_disc = True

    # image transformations
    transform = transforms.Compose([transforms.ToTensor()])

    train_set = datasets.MNIST(
        root='../input/data',
        train=True,
        download=True,
        transform=transform
    )
    test_set = datasets.MNIST(
        root='../input/data',
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False
    )

    device = get_device()
    make_dir()

    model = SparseAutoEncoder(x_dim=784, h_dim=500, z_dim=int(784 * 1.5), is_disc=is_disc).to(device)

    # the loss function
    criterion = nn.MSELoss()
    path_model = f"outputs/checkpoint/sparse_ae{epochs}.pth"  # _discrete_kl

    trainer = Trainer(model, criterion, learning_rate, device, add_sparsity, reg_param, num_enc_layer, num_dec_layer,
                      is_disc, is_kl, rho)
    # Load model
    model.load_state_dict(torch.load(path_model))
    
    enc_lab = pd.read_csv('embedding_labels_discrete_nokl.csv')
    
    group_lab = enc_lab.groupby('y')
    
    enc = []
    
    print('Grouping ...')
    for name, group in group_lab:
        enc.append(group.values[:, :-1])

    labels = 4
    idxz = 10
    
    z_in = torch.tensor(enc[labels][idxz], dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        z_in.to(device)
        
        rec = model.generate(z_in)

    plt.imshow(rec.resize(28, 28, 1), cmap='gray')
    plt.show()
