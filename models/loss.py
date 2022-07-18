import torch
import torch.nn.functional as F


# define the sparse loss function
def sparse_loss(auto_encoder, images):
    # get the layers as a list
    model_children = list(auto_encoder.children())

    loss = 0
    values = images
    for i in range(len(model_children)):
        values = F.relu((model_children[i](values)))
        loss += torch.mean(torch.abs(values))
    return loss
