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
    add_sparsity = 'yes'
    num_enc_layer = 3
    num_dec_layer = 3
    learning_rate = 0.001
    batch_size = 100
    is_disc = False

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
    path_model = f"outputs/checkpoint/sparse_ae{epochs}.pth"

    trainer = Trainer(model, criterion, learning_rate, device, add_sparsity, reg_param, num_enc_layer, num_dec_layer,
                      is_disc)
    trainer.train(train_loader, test_loader, epochs, path_model)

    # Load model
    model.load_state_dict(torch.load(path_model))

    encoding = None
    labels = None

    model.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader), total=int(len(test_loader) / test_loader.batch_size)):
            img, y = data
            img = img.to(device)
            img = img.view(img.size(0), -1)
            enc, _ = model(img)

            if i == 0:
                encoding = enc
                labels = y
            else:
                encoding = torch.cat((encoding, enc))
                labels = torch.cat((labels, y))

    encoding = encoding.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    col = ['coral', 'limegreen', 'royalblue', 'slateblue', 'orchid', 'palevioletred', 'chocolate', 'olive', 'palegreen',
           'teal']
    lab = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    targets = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    path = 'outputs/images/tsne.png'

    visualize_tsne(encoding, labels, col, lab, targets, path)
