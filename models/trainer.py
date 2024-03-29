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
        self.ANNEAL_RATE = 0.9995
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

        for _, data in tqdm(enumerate(dataloader), total=int(len(dataloader) / dataloader.batch_size)):
            counter += 1

            img, _ = data
            img = img.to(self.device)
            img = img.view(img.size(0), -1)

            self.optimizer.zero_grad()
            _, outputs = self.model(img, self.temperature)
            mse_loss = self.criterion(outputs, img)

            if self.add_sparsity == 'yes':
                if self.is_kl:
                    spars = sparse_loss_kl(self.model, img, self.rho, self.num_enc_layer, self.num_dec_layer,
                                         self.device, self.is_disc, self.temperature)
                else:
                    spars = sparse_loss(self.model, img, self.num_enc_layer, self.num_dec_layer, self.is_disc, 
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
        print(f"Train Loss: {loss:.3f}")
        print(f'Temperature: {self.temperature:.3f}')
        # save the reconstructed images every 5 epochs
        if epoch % 10 == 0 and self.dataset_name == 'MNIST':
            save_decoded_image(outputs.cpu(), f"outputs/images/train{epoch}.png")
        return epoch_loss

    # define the validation function
    def validate(self, dataloader, epoch):
        print('Validating')
        self.model.eval()
        running_loss = 0.0
        counter = 0

        with torch.no_grad():
            for _, data in tqdm(enumerate(dataloader), total=int(len(dataloader) / dataloader.batch_size)):
                counter += 1
                img, _ = data
                img = img.to(self.device)
                img = img.view(img.size(0), -1)
                _, outputs = self.model(img, self.temperature)
                loss = self.criterion(outputs, img)
                running_loss += loss.item()

                if counter % 100 == 1:
                    self.anneal_temp(lowerbound=5e-15)

        epoch_loss = running_loss / counter
        print(f"Val Loss: {loss:.3f}")
        print(f'Temperature: {self.temperature:.3f}')
        # save the reconstructed images every 5 epochs
        if epoch % 10 == 0 and self.dataset_name == 'MNIST':
            outputs = outputs.view(outputs.size(0), 1, 28, 28).cpu().data
            save_image(outputs, f"outputs/images/reconstruction{epoch}.png")

            n = min(img.size(0), 8)
            comparison = torch.cat([img.view(img.shape[0], 1, 28, 28)[:n],
                                    outputs.view(outputs.shape[0], 1, 28, 28)[:n]])
            save_image(comparison, f"outputs/images/comparison{epoch}.png")

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
