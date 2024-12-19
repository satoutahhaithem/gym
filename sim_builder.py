import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader, DistributedSampler
from torch.multiprocessing import Queue

import os
from tqdm import tqdm
import pandas as pd

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
        sampler = DistributedSampler(
            self.config.train_dataset, 
            num_replicas=self.config.num_nodes, 
            rank=self.rank, 
            shuffle=True, 
            drop_last=True
        )

        train_dataloader = DataLoader(self.config.train_dataset, 
                          batch_size=self.config.batch_size,
                          sampler=sampler)

        val_dataloader = DataLoader(self.config.val_dataset, 
                          batch_size=self.config.batch_size,
                          shuffle=True)

        return train_dataloader, val_dataloader

    def _process_cleanup(self):
        dist.destroy_process_group()

    def _execute(self, rank, queue):
        self.rank = rank

        self._process_setup()
        self.train_dataloader, self.val_dataloader = self._build_dataloaders()

        sim = TrainNode(self.config,
                  self.train_dataloader,
                  self.val_dataloader,
                  self.device,
                  self.rank)

        
        # Capture losses during training
        val_losses, train_losses = sim.train(epochs=self.config.num_epochs)

        # Send metrics to the main process
        queue.put({'rank': self.rank, 'val_losses': val_losses, 'train_losses': train_losses})

        self._process_cleanup()
        

    def execute(self):
        queue = Queue()

        torch.multiprocessing.spawn(self._execute, args=(queue,), nprocs=self.config.num_nodes, join=True)

        metrics = []
        while not queue.empty():
            metrics.append(queue.get())

        pd.DataFrame({
            f'rank{x['rank']}trainloss':x['train_losses'] for x in metrics
        }).to_parquet('log/train_loss.pq')

        pd.DataFrame({
            f'rank{x['rank']}valloss':x['val_losses'][0] for x in metrics
        }).to_parquet('log/val_loss.pq')

        pd.DataFrame({
            f'rank{x['rank']}valacc':x['val_losses'][1] for x in metrics
        }).to_parquet('log/val_accuracy.pq')