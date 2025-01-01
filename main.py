import os
import time

import torch
import torch.nn as nn

from sim_builder import *
from sim_config import *


class ToyModel(torch.nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()  # Add sigmoid for binary classification

    def forward(self, x):
        return self.sigmoid(self.net2(self.relu(self.net1(x))))


class DotProductDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=1000, threshold=0.0):
        super(DotProductDataset, self).__init__()
        self.data = torch.randn(num_samples, 10)  # Random 10-dimensional inputs
        self.weights = torch.randn(10)           # Fixed weight vector
        self.threshold = threshold
        # Labels are 1 if dot product > threshold, otherwise 0
        self.targets = (self.data @ self.weights > self.threshold).float().unsqueeze(1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]



def setup_config():
    config = SimConfig()

    config.model_class = ToyModel
    config.train_dataset = DotProductDataset(num_samples=10000)
    config.val_dataset = DotProductDataset(num_samples=100)

    config.optimizer_class = torch.optim.SGD
    config.optimizer_kwargs = {
            'lr': 0.01
    }
    config.criterion_class = torch.nn.MSELoss

    config.num_nodes = 4

    return config

def main():
    config = setup_config()

    simbuilder = LocalSimBuilder(config)

    simbuilder.execute()

if __name__ == "__main__":
    os.environ['VERBOSITY'] = '1'
    main()
