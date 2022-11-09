import time
from tqdm import tqdm
import torch
from torchvision.utils import save_image

from models.loss import sparse_loss, sparse_loss_kl
from utils.utility import save_decoded_image
from utils.plotter import plot_loss

import numpy as np


class Trainer:
    def __init__(self, model, criterion, params):

        self.model = model
        self.criterion = criterion
        self.params = params
        self.learning_rate = params['learning_rate']
        self.device = params['device']
        self.add_sparsity = params['add_sparsity']
        self.reg_param = params['reg_param']
        self.num_enc_layer = params['num_enc_layer']
        self.num_dec_layer = params['num_dec_layer']
        self.is_disc = params['is_disc']
        self.is_kl = params['is_kl']
        self.rho = params['rho']
        self.dataset_name = params['dataset']
        self.temperature = 1.
        self.ANNEAL_RATE = 0.995
        self.temp_min = 0.5

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def anneal_temp(self, lowerbound=1e-5):
        if self.temperature > lowerbound:
            self.temperature = self.temperature * self.ANNEAL_RATE

    # define the training function
    def train_step(self, dataloader, epoch):
        print('Training')
        self.model.train()
        running_loss = 0.0
        counter = 0
        loss = 0

        for _, x in tqdm(enumerate(dataloader), total=int(len(dataloader) / dataloader.batch_size)):
            counter += 1
            
            x = x.to(self.device)

            self.optimizer.zero_grad()
            _, outputs, _ = self.model(x, self.temperature)
            mse_loss = self.criterion(outputs, x)

            if self.add_sparsity == 'yes':
                if self.is_kl:
                    spars = sparse_loss_kl(self.model, x, self.rho, self.num_enc_layer, self.num_dec_layer,
                                           self.device, self.is_disc, self.temperature)
                else:
                    spars = sparse_loss(self.model, x, self.num_enc_layer, self.num_dec_layer, self.is_disc, 
                                        self.temperature)
                # add the sparsity penalty
                loss = mse_loss + self.reg_param * spars
            else:
                loss = mse_loss

            loss.backward()
            self.optimizer.step()

            if counter % 100 == 1:
                self.anneal_temp(lowerbound=5e-15)

            running_loss += loss.item()

        epoch_loss = running_loss / counter
        print()
        print(f"Train Loss: {epoch_loss:.3f}")
        print()

        return epoch_loss

    # define the validation function
    def validate(self, dataloader, epoch):
        print('Validating')
        self.model.eval()
        running_loss = 0.0
        counter = 0

        with torch.no_grad():
            for _, x in tqdm(enumerate(dataloader), total=int(len(dataloader) / dataloader.batch_size)):
                counter += 1
                
                x = x.to(self.device)
                
                _, outputs, _ = self.model(x, self.temperature)
                loss = self.criterion(outputs, x)
                running_loss += loss.item()

                if counter % 100 == 1:
                    self.anneal_temp(lowerbound=5e-15)

        epoch_loss = running_loss / counter
        print()
        print(f"Val Loss: {epoch_loss:.3f}")
        print()

        return epoch_loss

    def train(self, train_loader, test_loader, epochs, path_model):
        # train and validate the autoencoder neural network
        train_loss = []
        val_loss = []
        start = time.time()
        self.temperature = 1.
        try:
            for epoch in range(epochs):
                print(f"Epoch {epoch + 1} of {epochs}")
                train_epoch_loss = self.train_step(train_loader, epoch)
                val_epoch_loss = self.validate(test_loader, epoch)
                train_loss.append(train_epoch_loss)
                val_loss.append(val_epoch_loss)
        except KeyboardInterrupt:
            print('.'*20)
        end = time.time()

        print(f"{(end - start) / 60:.3} minutes")
        # save the trained model
        torch.save(self.model.state_dict(), path_model)

        plot_loss(train_loss, val_loss, f"outputs/loss_{self.dataset_name}.png")
