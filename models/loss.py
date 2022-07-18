import torch
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
        values = fc_layer(values)

    loss += torch.mean(torch.abs(values))

    # Decoding
    for i in range(num_dec_layer-1):
        fc_layer = list(auto_encoder.decoder.children())[0][2 * i]
        relu = list(auto_encoder.decoder.children())[0][2 * i + 1]
        values = relu(fc_layer(values))
        loss += torch.mean(torch.abs(values))
    
    return loss
