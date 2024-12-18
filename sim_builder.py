import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader, DistributedSampler

import os
from tqdm import tqdm

from sim_config import *
from train_node import *

class SimBuilder:
    def __init__(self, 
                 config: SimConfig):
        self.config = config

    def _process_setup(self):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        # initialize the process group
        # TODO: doesn't have to be gloo & cpu
        dist.init_process_group("gloo", rank=self.rank, world_size=self.config.num_nodes)
        self.device = torch.device("cpu")

    def _build_dataloaders(self):
        # TODO: distributed dataloader 
        train_dataloader = DataLoader(self.config.train_dataset, 
                          batch_size=self.config.batch_size,
                          shuffle=True)
        val_dataloader = DataLoader(self.config.val_dataset, 
                          batch_size=self.config.batch_size,
                          shuffle=True)

        return train_dataloader, val_dataloader

    def _process_cleanup(self):
        dist.destroy_process_group()


    def _execute(self, rank):
        self.rank = rank

        self._process_setup()
        self.train_dataloader, self.val_dataloader = self._build_dataloaders()

        sim = TrainNode(self.config,
                  self.train_dataloader,
                  self.val_dataloader,
                  self.device,
                  self.rank)
        sim.train(epochs=100)

        self._process_cleanup()
        

    def execute(self):
        torch.multiprocessing.spawn(self._execute, args=(), nprocs=self.config.num_nodes, join=True)
