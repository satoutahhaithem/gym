import torch
from torch.utils.data import DataLoader, DistributedSampler

import os
from tqdm import tqdm
import pandas as pd

from .sim_config import *
from .train_node import *
from .communication_handler import *

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
        model_clone = self.config.model_class(self.config.gpt_config).to(self.config.device)

        for name, param in model_clone.named_parameters():
            reduced_data = torch.zeros_like(param.data)

            for rank in range(self.config.num_nodes):
                if name in self.train_nodes[rank].model.state_dict():
                    reduced_data += self.train_nodes[rank].model.state_dict()[name]
                else:
                    raise ValueError(f'{name} not found in buffer')

            param.data.copy_(reduced_data / self.config.num_nodes)

        for model, model_name in zip([model_clone, self.train_nodes[0].model], 
                               ['val_global', 'val_local']):
            model.eval()

            loss_total = 0

            with torch.no_grad():
                for _ in range(int(self.config.val_size / self.config.batch_size)):
                    x, y = self.train_nodes[0]._get_batch(eval=True)
                    
                    output = model(x).transpose(1, 2)
                    loss = self.train_nodes[0].criterion(output, y)

                    loss_total += loss.item()

                self.logger.log_pure(loss=loss_total / int(self.config.val_size / self.config.batch_size), 
                                     name=model_name)

        del model_clone

    def execute(self):
        for rank in range(self.config.num_nodes):
            self.train_nodes.append(TrainNode(self.config, 
                                              torch.device(self.config.device), 
                                              rank, 
                                              self.logger,
                                              self.communication_handler,
                                              state_dict=None if rank == 0 else self.train_nodes[0].model.state_dict()))

        while self.local_step < self.max_steps:
            if self.local_step % self.config.eval_interval == 0:
                self._evaluate_step()

            self._train_step()

            self.local_step += 1
            self.logger.increment_step()

        self._evaluate_step()
