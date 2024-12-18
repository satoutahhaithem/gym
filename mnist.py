import os
import time

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from sim_builder import *
from sim_config import *

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # Output: 16x28x28
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # Output: 32x28x28
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 32x14x14
        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def setup_config():
    config = SimConfig()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to range [-1, 1]
    ])

    config.train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    config.val_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    config.model_class = CNN

    config.optimizer_class = torch.optim.Adam
    config.optimizer_kwargs = {
            'lr': 0.01
    }
    config.criterion_class = torch.nn.CrossEntropyLoss

    config.num_epochs = 5

    config.num_nodes = 4

    return config

def main():
    config = setup_config()

    simbuilder = SimBuilder(config)

    simbuilder.execute()

if __name__ == "__main__":
    os.environ['VERBOSITY'] = '1'
    main()
