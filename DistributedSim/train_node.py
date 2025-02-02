import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn.functional as F

from typing import Optional, Callable, Type
import os
import copy

from .sim_config import *
from .gradient_strategy import *
from .wandb_logger import *

from tqdm import tqdm

class TrainNode:
    '''
    Single node of distributed training process. Should be the same regardless of rank topology/architecture.
    '''
    def __init__(self, 
                 config: SimConfig,
                 device: torch.device,
                 rank: int):
        self.config = config

        self.device = device
        self.rank = rank

        self.model = self.config.model_class(self.config.gpt_config).to(self.device)
        
        ## Ensure all process models share the same params
        if self.config.num_nodes > 1:
            for _, param in self.model.named_parameters():
                dist.broadcast(param.data, src=0)

        self.build_dataloaders()

        self.criterion = self.config.criterion_class(**self.config.criterion_kwargs)

        self.local_step = 0
        self.max_steps = len(self.train_dataloader) * self.config.num_epochs

        # if self.rank == 0:
        self.logger = WandbLogger(rank=self.rank, 
                                  device=self.device, 
                                  config=self.config, 
                                  model=self.model, 
                                  max_steps=self.max_steps, 
                                  project=self.config.wandb_project)

        self.gradient_strategy = self.config.gradient_class(self.rank, 
                                                            self.model, 
                                                            self.config,
                                                            self.logger if self.rank == 0 else None)

        self.train_data_iter = iter(self.train_dataloader)
        self.val_data_iter = iter(self.val_dataloader)
        
    
    def build_dataloaders(self):
        sampler = DistributedSampler(
            self.config.train_dataset, 
            num_replicas=self.config.num_nodes, 
            rank=self.rank, 
            shuffle=True, 
            drop_last=True
        )

        self.train_dataloader = DataLoader(self.config.train_dataset, 
                          batch_size=self.config.batch_size,
                          sampler=sampler)

        self.val_dataloader = DataLoader(self.config.val_dataset, 
                          batch_size=self.config.batch_size,
                          shuffle=True)

    def _save_checkpoint(self):
        if not os.path.exists(os.path.join(self.config.save_dir, self.config.wandb_project, self.logger.wandb_run_id, str(self.rank))):
            os.makedirs(os.path.join(self.config.save_dir, self.config.wandb_project, self.logger.wandb_run_id, str(self.rank)), exist_ok=True)

        filename = f"{self.local_step}.pt"
        torch.save(self.model.state_dict(), os.path.join(self.config.save_dir, self.config.wandb_project, self.logger.wandb_run_id, str(self.rank), filename))

    def _get_batch(self, eval=False):
        if not eval or self.val_data_iter is None:
            try:
                x, y = next(self.train_data_iter)
            except StopIteration:
                self.epoch += 1
                self.train_data_iter = iter(self.train_dataloader)
                x, y = next(self.train_data_iter)
        else:
            try:
                x, y = next(self.val_data_iter)
            except StopIteration:
                self.val_data_iter = iter(self.val_dataloader)
                x, y = next(self.val_data_iter)

        x, y = x.to(self.device), y.to(self.device)

        return x, y

    def _train_step(self):
        x, y = self._get_batch()
        
        self.gradient_strategy.zero_grad()

        output = self.model(x).transpose(1, 2)
        loss = self.criterion(output, y)
        loss.backward()
        self.gradient_strategy.step()

        if self.rank == 0:
            self.logger.log_train(loss=loss.item())

        if self.local_step % self.config.checkpoint_interval == 0:
            self._save_checkpoint()

        return loss.item()

    def _evaluate(self):
        model_clone = self.config.model_class(self.config.gpt_config)
        model_clone.load_state_dict(copy.deepcopy(self.model.state_dict()))

        for name, param in model_clone.named_parameters():
            dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
            param.data = param.data / dist.get_world_size()

        if self.rank == 0:
            # For rank 0, we will calculate the local loss
            self.model.eval()

            with torch.no_grad():
                x, y = self._get_batch(eval=True)
                
                output = self.model(x).transpose(1, 2)
                loss = self.criterion(output, y)

                self.logger.log_pure(loss=loss.item(), name="val_local")

        if self.rank == 1:
            # For rank 1, we want to calculate the average model loss
            model_clone.eval()

            print('calculating global loss')

            with torch.no_grad():   
                x, y = self._get_batch(eval=True)
                
                output = model_clone(x).transpose(1, 2)
                loss = self.criterion(output, y)

                print(f'loss: {loss.item()}')

                self.logger.log_pure(loss=loss.item(), name="val_global")

            del model_clone

    def train(self):
        while self.local_step < self.max_steps:
            if self.local_step % self.config.eval_interval == 0:
                self._evaluate()

            self._train_step()

            self.local_step += 1
            self.logger.increment_step()