import torch.nn as nn
from utils.utility import gumbel_sigmoid


class Encoder(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super(Encoder, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        self.neurons = [self.x_dim, *self.h_dim]
        self.hidden_layers = nn.ModuleList([nn.Linear(self.neurons[i - 1], self.neurons[i])
                                            for i in range(1, len(self.neurons))])
        self.output = nn.Linear(self.h_dim[-1], self.z_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        for h in self.hidden_layers:
            x = self.relu(h(x))
        return self.output(x)


class Decoder(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super(Decoder, self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        self.neurons = [self.x_dim, *self.h_dim]
        self.hidden_layers = nn.ModuleList([nn.Linear(self.neurons[i - 1], self.neurons[i])
                                            for i in range(1, len(self.neurons))])
        self.output = nn.Linear(self.h_dim[-1], self.z_dim)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for h in self.hidden_layers:
            x = self.relu(h(x))
        return self.sigmoid(self.output(x))


class SparseAutoEncoder(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, is_disc=False):
        super(SparseAutoEncoder, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.is_disc = is_disc

        self.encoder = Encoder(self.x_dim, self.h_dim, self.z_dim)
        self.decoder = Decoder(self.z_dim, self.h_dim, self.x_dim)

    def forward(self, x, temperature=1.):
        enc = self.encoder(x)

        if self.is_disc:
            enc = gumbel_sigmoid(enc, temperature)

        rec = self.decoder(enc)

        return enc, rec
