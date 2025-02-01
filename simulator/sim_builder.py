import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader, DistributedSampler
from torch.multiprocessing import Queue

import os
from tqdm import tqdm
import pandas as pd

from .sim_config import *
from .train_node import *


class SimBuilder:
    '''
    SimBuilder is used to spawn processes and connect them together for distributed training.
    Spawns multiple TrainNode instances. TrainNode should be the same no matter rank topology/architecture.
    '''
    def __init__(self, 
                 config: SimConfig):
        self.config = config

    @abstractmethod
    def _build_connection(self):
        raise NotImplementedError

    def _process_cleanup(self):
        dist.destroy_process_group()

    def _execute(self, rank, queue):
        self.rank = rank

        self._build_connection()

        sim = TrainNode(self.config,
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

        train_loss_series = pd.DataFrame({
            f'rank{x['rank']}trainloss':x['train_losses'] for x in metrics
        }).mean(axis=1)

        val_loss_series = pd.DataFrame({
            f'rank{x['rank']}valloss':x['val_losses'][0] for x in metrics
        }).mean(axis=1)

        val_accuracy_series = pd.DataFrame({
            f'rank{x['rank']}valacc':x['val_losses'][1] for x in metrics
        }).mean(axis=1)

        return train_loss_series, val_loss_series, val_accuracy_series

class LocalSimBuilder(SimBuilder):
    def _build_connection(self):
        '''
        This is the default callback for setting up pytorch distributed connections.
        All ranks are assumed to be on the same machine, and device is defaulted to cpu.
        '''
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        # initialize the process group
        # TODO: doesn't have to be gloo & cpu
        dist.init_process_group("gloo", rank=self.rank, world_size=self.config.num_nodes)
        self.device = torch.device("cpu")