import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group

from typing import Optional, Callable, Type
import os

from sim_config import *

class TrainNode:
    def __init__(self, 
                 config: SimConfig,
#                  model_class: Type[torch.nn.Module],
                 dataloader: torch.utils.data.DataLoader,
#                  optimizer: Type[torch.optim.Optimizer],
                 device: torch.device,
                 rank: int):
        self.config = config

        self.dataloader = dataloader
        self.device = device
        self.rank = rank

        self.model = self.config.model_class().to(self.device) # TODO: model kwargs
        ## Ensure all process models share the same params
        for _, param in self.model.named_parameters():
            dist.broadcast(param.data, src=0)

        self.optimizer = self.config.optimizer_class(self.model.parameters(),
                                                     **self.config.optimizer_kwargs)

        self.criterion = self.config.criterion_class(**self.config.criterion_kwargs)

        # for _, param in self.model.named_parameters():
        #     print(f'Process {self.rank} params {param}')
        #     break

    def train_epoch(self):
        self.model.train()

        if int(os.environ['VERBOSITY']) >= 3:
            print(f'Process {self.rank} train')

        loss_sum = 0

        for i, batch in enumerate(self.dataloader, 0):
            self.optimizer.zero_grad()

            X, y = batch
            yhat = self.model.forward(X)

            loss = self.criterion(y, yhat)

            # TODO: AllReduce is currently doing the default - summing gradients (high bandwidth cost) 
            loss.backward()
            dist.barrier()

            for param in self.model.parameters():
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= dist.get_world_size()
            
            self.optimizer.step()

            for name, param in self.model.named_parameters():
#                 print(f'\nProcess {self.rank} {name} gradient {param.grad}')
                break

            loss_sum += loss.item()

        print(f'\n Process {self.rank} loss {loss_sum}')

        dist.barrier()


    def train(self, epochs=10):
        for epoch in range(epochs):
            self.train_epoch()
