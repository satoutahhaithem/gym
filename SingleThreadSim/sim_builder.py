import torch
from torch.utils.data import DataLoader, DistributedSampler

import os
from tqdm import tqdm
import pandas as pd

from .sim_config import *
from .train_node import *


class SingleThreadSimBuilder:
    '''
    SimBuilder is used to spawn processes and connect them together for distributed training.
    Spawns multiple TrainNode instances. TrainNode should be the same no matter rank topology/architecture.
    '''
    def __init__(self, 
                 config: SimConfig):
        self.config = config

        self.local_step = 0
        self.max_steps = int(len(self.config.train_dataset) \
                         / (self.config.batch_size * self.config.num_nodes) \
                         * self.config.num_epochs)
        if self.config.gradient_config.max_local_steps:
            self.max_steps = min(self.max_steps, 
                                 self.config.gradient_config.max_local_steps)

        self.logger = WandbLogger(config=self.config, 
                                  max_steps=self.max_steps, 
                                  project=self.config.wandb_project)

        self.communication_handler = CommunicationHandler(self.config)

        self.train_nodes = []

    def _train_step(self):
        for rank in range(self.config.num_nodes):
            self.train_nodes[rank].train_step()

            if self.local_step % self.config.checkpoint_interval == 0:
                self.train_nodes[rank].save_checkpoint(self.local_step)

    def _evaluate_step(self):
        for rank in range(self.config.num_nodes):
            self.train_nodes[rank].evaluate_step()

    def execute(self):
        for rank in range(self.config.num_nodes):
            self.train_nodes.append(TrainNode(self.config, 
                                              torch.device(self.config.device), 
                                              rank, 
                                              self.logger,
                                              self.communication_handler))

        while self.local_step < self.max_steps:
            # if self.local_step % self.config.eval_interval == 0:
            #     self._evaluate_step()

            self._train_step()

            self.local_step += 1
            self.logger.increment_step()

        # self._evaluate_step()
