import os
import torch
from torchvision.utils import save_image


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
    u = torch.rand_like(logits)
    y = logits + torch.log(u) - torch.log(1 - u)

    return torch.sigmoid(y / temperature)


def gumbel_sigmoid(logits, temperature=1e-5):
    y = gumbel_sigmoid_sample(logits, temperature)
    y_hard = (y > .5).float()

    return (y_hard - y).detach() + y

