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
            nn.Linear(self.h_dim, self.x_dim)
            # nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)


class SparseAutoEncoder(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, is_disc=False, rec_disc=False):
        super(SparseAutoEncoder, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.is_disc = is_disc
        self.rec_disc = rec_disc

        self.encoder = Encoder(self.x_dim, self.h_dim, self.z_dim)
        self.decoder = Decoder(self.x_dim, self.h_dim, self.z_dim)
        self.l_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        
    def generate(self, z):
        return self.decoder(z)

    def forward(self, x, temperature=1.):
        enc = self.encoder(x)  # logits

        z_hard = self.sigmoid(enc)
        z_hard = (z_hard > .5).float()

        if self.is_disc:
            enc = gumbel_sigmoid(enc, temperature)
        else:
            enc = self.l_relu(enc)
            
        rec = self.decoder(enc)

        if self.rec_disc:
            rec = gumbel_sigmoid(rec, temperature)
        else:
            rec = self.sigmoid(rec)

        return enc, rec, z_hard
