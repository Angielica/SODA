import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utility import gumbel_sigmoid


# define the sparse loss function
def sparse_loss(auto_encoder, images, num_enc_layer, num_dec_layer, is_disc=False, temperature=1.):
    loss = 0
    values = images

    # Encoding
    for i in range(num_enc_layer-1):
        fc_layer = list(auto_encoder.encoder.children())[0][2 * i]
        relu = list(auto_encoder.encoder.children())[0][2 * i + 1]
        values = relu((fc_layer(values)))
        loss += torch.mean(torch.abs(values))

    fc_layer = list(auto_encoder.encoder.children())[0][-1]

    if is_disc:
        values = gumbel_sigmoid(fc_layer(values), temperature)
    else:
        values = nn.LeakyReLU()(fc_layer(values))

    loss += torch.mean(torch.abs(values))

    # Decoding
    for i in range(num_dec_layer-1):
        fc_layer = list(auto_encoder.decoder.children())[0][2 * i]
        relu = list(auto_encoder.decoder.children())[0][2 * i + 1]
        values = relu(fc_layer(values))
        loss += torch.mean(torch.abs(values))
    
    return loss


def kl_divergence(p, p_hat, device):
    sig = nn.Sigmoid()
    p_hat = torch.mean(sig(p_hat), 1)
    p_tensor = torch.Tensor([p] * len(p_hat)).to(device)
    kl = torch.sum(p_tensor * torch.log(p_tensor) - p_tensor * torch.log(p_hat) +
                   (1 - p_tensor) * torch.log(1 - p_tensor) - (1 - p_tensor) * torch.log(1 - p_hat))
    return kl


def sparse_loss_kl(auto_encoder, images, rho, num_enc_layer, num_dec_layer, device, is_disc=False, temperature=1.):
    loss = 0
    values = images

    # Encoding
    for i in range(num_enc_layer - 1):
        fc_layer = list(auto_encoder.encoder.children())[0][2 * i]
        values = fc_layer(values)
        loss += kl_divergence(rho, values, device)

    fc_layer = list(auto_encoder.encoder.children())[0][-1]

    if is_disc:
        values = gumbel_sigmoid(fc_layer(values), temperature)  # ?
    else:
        values = fc_layer(values)

    loss += kl_divergence(rho, values, device)

    # Decoding
    for i in range(num_dec_layer - 1):
        fc_layer = list(auto_encoder.decoder.children())[0][2 * i]
        values = fc_layer(values)
        loss += kl_divergence(rho, values, device)

    return loss

