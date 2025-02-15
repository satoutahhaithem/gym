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

    def _execute(self, rank):
        self.rank = rank

        self._build_connection()

        sim = TrainNode(self.config,
                  self.device,
                  self.rank)
        
        sim.train()

        self._process_cleanup()
        

    def execute(self):
        torch.multiprocessing.spawn(self._execute, args=(), nprocs=self.config.num_nodes, join=True)

class LocalSimBuilder(SimBuilder):
    def _build_connection(self):
        '''
        This is the default callback for setting up pytorch distributed connections.
        All ranks are assumed to be on the same machine, and device is defaulted to cpu.
        '''
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(12355 + self.config.gpu_offset - (10 if self.config.device == 'cpu' else 0))

        # initialize the process group
        if self.config.device_type == 'cuda':
            dist.init_process_group("nccl" if len(self.config.devices) == self.config.num_nodes else "gloo", 
                                    rank=self.rank, 
                                    world_size=self.config.num_nodes)
            self.device = torch.device(f"cuda:{self.config.devices[self.rank % len(self.config.devices)]}")
            torch.cuda.set_device(self.device)
        elif self.config.device_type == 'cpu':
            dist.init_process_group("gloo", 
                                    rank=self.rank, 
                                    world_size=self.config.num_nodes)
            self.device = torch.device("cpu")
        elif self.config.device_type == 'mps':
            dist.init_process_group("gloo", 
                                    rank=self.rank, 
                                    world_size=self.config.num_nodes)
            self.device = torch.device("mps")
        else:
            raise ValueError(f"Invalid device type: {self.config.device}")

        print(f"Rank {self.rank} using device {self.device}")

class SingleSimBuilder(SimBuilder):
    def _build_connection(self):
        self.device = torch.device(self.config.device)