import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning import Trainer

# Hyper Parameters
input_size = 784 # 28x28
hidden_size = 500
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

class LitNeuralNet(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LitNeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)

        return out

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        images, labels = batch

        # 100, 1, 28, 28
        # 100, 784
        images = images.reshape(-1, 28 * 28)

        # forward
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)

        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = learning_rate)

    def train_dataloader(sel):
        train_dataset = torchvision.datasets.MNIST(root = './data', train = True, transform = transforms.ToTensor(), download = True)
        train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, num_workers = 4, shuffle = True)
        return train_loader

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        images, labels = batch

        # 100, 1, 28, 28
        # 100, 784
        images = images.reshape(-1, 28 * 28)

        # forward
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)

        return {'val_loss': loss}

    def val_dataloader(sel):
        val_dataset = torchvision.datasets.MNIST(root = './data', train = False, transform = transforms.ToTensor(), download = False)
        val_loader = torch.utils.data.DataLoader(dataset = val_dataset, batch_size = batch_size, num_workers = 4, shuffle = False)
        return val_loader

if __name__ == '__main__':
    trainer = Trainer( max_epochs = num_epochs, fast_dev_run = False)
    model = LitNeuralNet(input_size, hidden_size, num_classes)
    trainer.fit(model)
