import torch.nn as nn
from utils.utility import gumbel_sigmoid


class Encoder(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super(Encoder, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Linear(self.x_dim, self.h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.h_dim, self.z_dim)
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super(Decoder, self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        
        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, self.h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.h_dim, self.x_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)


class SparseAutoEncoder(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, is_disc=False):
        super(SparseAutoEncoder, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.is_disc = is_disc

        self.encoder = Encoder(self.x_dim, self.h_dim, self.z_dim)
        self.decoder = Decoder(self.x_dim, self.h_dim, self.z_dim)

    def forward(self, x, temperature=1.):
        enc = self.encoder(x)  # logits

        if self.is_disc:
            enc = gumbel_sigmoid(enc, temperature)

        rec = self.decoder(enc)  # sigmoid

        return enc, rec