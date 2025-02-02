import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn.functional as F

from typing import Optional, Callable, Type
import os

from .sim_config import *
from .gradient_strategy import *

from tqdm import tqdm

import wandb

class WandbLogger:
    def __init__(self, config: SimConfig, max_steps: int, project: str):
        self.config = config
        self.project = project

        self.pbar = tqdm(total=max_steps)

        self.step = 0

        wandb.init(
            # set the wandb project where this run will be logged
            project=self.config.wandb_project,

            # track hyperparameters and run metadata
            config={
                "learning_rate": self.config.optimizer_kwargs['lr'],
                "architecture": "GPT",
                # "dataset": self.config.train_dataset,
                "epochs": self.config.num_epochs,
            }
        )

    def log_train(self, loss: float):
        wandb.log({"train_loss": loss}, step=self.step)

        self.pbar.update(1)
        self.pbar.set_postfix(
            {
                "loss": f"{loss:.4f}",
                # "lr": f"{lr:.4f}",
            }
        )

        self.step += 1


    def log_val(self, loss: float):
        wandb.log({"val_loss": loss}, step=self.step)

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

        self.criterion = self.config.criterion_class(**self.config.criterion_kwargs)

        self.gradient_strategy = self.config.gradient_class(self.model, self.config)

        self.build_dataloaders()

        self.local_step = 0
        self.max_steps = len(self.train_dataloader) * self.config.num_epochs

        self.train_data_iter = iter(self.train_dataloader)
        self.val_data_iter = iter(self.val_dataloader)
        
        if self.rank == 0:
            self.logger = WandbLogger(config=self.config, max_steps=self.max_steps, project=self.config.wandb_project)
            

    
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

    # def _save_checkpoint(self):
    #     torch.save(self.model.state_dict(), os.path.join(self.config.save_dir, f"model_{self.epoch}.pt"))

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

        return loss.item()

    def _evaluate(self):
        self.model.eval()
        
        with torch.no_grad():
            x, y = self._get_batch(eval=True)
            
            output = self.model(x).transpose(1, 2)
            loss = self.criterion(output, y)
            
            # Synchronize loss across processes
            if self.config.num_nodes > 1:
                dist.barrier()
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                loss = loss / dist.get_world_size()
            
            if self.rank == 0:
                self.logger.log_val(loss=loss.item())
            
            # if self.rank == 0:
            #     print(loss.item())

            return loss.item()

    def train(self):
        while self.local_step < self.max_steps:
            if self.local_step % self.config.eval_interval == 0:
                self._evaluate()

            loss = self._train_step()

            self.local_step += 1
