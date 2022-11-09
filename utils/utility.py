import os
import torch
from torchvision.utils import save_image

from tqdm import tqdm
import pandas as pd
from matplotlib import colors
import numpy as np
import random


def data_mapping(model, loader, params):
    model.eval()
    z_list = []
    rec_list = []

    with torch.no_grad():
        for i, data in enumerate(loader):
            data, _ = data
            data = data.to(params['device'])
            z, recon_batch = model(data)
            z_list.append(z)
            rec_list.append(recon_batch)

    res_z = np.array(torch.vstack(z_list).detach().cpu())
    tmp = np.array(torch.vstack(rec_list).detach().cpu())

    res_rec = np.zeros_like(tmp, dtype=int)
    res_rec[tmp > 0.5] = 1

    return res_z, res_rec


def create_color_map(params, dataset):
    color_dict = dict(colors.CSS4_COLORS)
    del color_dict['whitesmoke']
    del color_dict['white']
    del color_dict['snow']
    del color_dict['seashell']
    del color_dict['linen']
    del color_dict['floralwhite']
    del color_dict['cornsilk']
    del color_dict['ivory']
    del color_dict['beige']
    del color_dict['lightyellow']
    del color_dict['lightgoldenrodyellow']
    del color_dict['honeydew']
    del color_dict['mintcream']
    del color_dict['azure']
    del color_dict['lightcyan']
    del color_dict['aliceblue']
    del color_dict['ghostwhite']
    del color_dict['lavender']
    del color_dict['lavenderblush']

    np.random.seed(1011)
    unique_colors = np.random.choice(
        list(color_dict.keys()), params['n_itemsets'], False)
    color_map = dict(zip(dataset['chosen_itemsets'],
                         [unique_colors[k] for k in list(range(params['n_itemsets']))]))

    color_list = [color_map[k] for k in dataset['test_ids']]
    return color_map, color_list


# get the computation device
def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device


# make the `images` directory
def make_dir():
    image_dir = 'outputs/images'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)


# for saving the reconstructed images
def save_decoded_image(img, name):
    img = img.view(img.size(0), 1, 28, 28)
    save_image(img, name)


def gumbel_sigmoid_sample(logits, temperature=1.):
    # u = torch.rand_like(logits)
    # y = logits + torch.log(u) - torch.log(1 - u)

    # dev = torch.fill(torch.zeros_like(u), 1. / torch.sqrt(torch.tensor(12.)))
    # beta = 0.78 * dev
    # mu = beta * torch.log(torch.log(torch.tensor(2)))
    # E = mu + 0.557721 * beta

    y = logits

    return torch.sigmoid(y / temperature)  # , torch.sigmoid(E / temperature)


def gumbel_sigmoid(logits, temperature=1e-5):
    y = gumbel_sigmoid_sample(logits, temperature)
    y_hard = (y > .5).float()
    # E_y_hard = (E_y > .5).float()

    return (y_hard - y).detach() + y  # , (E_y_hard - E_y).detach() + E_y


def testing(model, test_loader, params):
    dataset_name = params['dataset']
    device = params['device']
    encoding, labels = None, None

    model.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader), total=int(len(test_loader) / test_loader.batch_size)):
            img, y = data
            img = img.to(device)
            # x = x.view(x.size(0), -1)
            _, _, enc = model(img)

            if i == 0:
                encoding = enc
                labels = y
            else:
                encoding = torch.cat((encoding, enc))
                labels = torch.cat((labels, y))

    encoding = encoding.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    df = pd.DataFrame(encoding)
    df['y'] = labels

    print('*' * 20)
    print(df)
    print('*' * 20)
    df.to_csv(f'embedding_{dataset_name}.csv', index=False)

    return encoding, labels

