import wandb
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import os

from .sim_config import *
from .utils import extract_config, create_config


class Logger:
  def __init__(self,
               model: nn.Module,
               max_steps: int):
    self.model = model
    self.max_steps = max_steps

    self.pbar = tqdm(total=self.max_steps, initial=0)

    print(f'Logger initialized.')

    self.step = 0
    self.current_lr = 0
    
  def log(self, data: dict):
    if hasattr(self, 'wandb_name'):
      wandb.log(data, step=self.step)

  def log_loss(self, loss: float, name: str):
    if hasattr(self, 'wandb_name'):
      data = {
        f"{name}_loss": loss,
        f"{name}_perplexity": float(np.exp(loss))
      }
      wandb.log(data, step=self.step)

  def log_train(self, loss: float):
    if hasattr(self, 'wandb_name'):
      data = {
        "train_loss": loss,
        "train_perplexity": float(np.exp(loss)),
      }
      if self.current_lr:
        data["lr"] = self.current_lr

      wandb.log(data, step=self.step)

    self.pbar.update(1)
    self.pbar.set_postfix({
      "train_loss": f"{loss:.4f}",
      "lr": f"{self.current_lr:.6f}",
    })

  def increment_step(self):
    self.step += 1

  def log_lr(self, lr: float):
    self.current_lr = lr


class WandbLogger(Logger):
  def __init__(self,
               model: nn.Module,
               max_steps: int,
               strategy=None,
               train_node=None,
               wandb_project: str = None,
               wandb_name: str = None):
    super().__init__(model, max_steps)
    
    self.wandb_project = wandb_project
    self.wandb_name = wandb_name or None

    # Create wandb configuration using the utility function
    wandb_config = create_config(
      model=model,
      strategy=strategy,
      train_node=train_node,
      extra_config={
        "max_steps": max_steps,
      }
    )

    init_kwargs = {
      "project": self.wandb_project,
      "name": self.wandb_name,
      "config": wandb_config,
      "resume": "allow" # Allow resuming if possible, or create new
    }

    wandb.init(**init_kwargs)
    
    # Set the logger's step based on wandb's step for the run
    print(f"Started new wandb run '{self.wandb_name}' (ID: {wandb.run.id}). Starting at step {self.step}.")

    # Update tqdm progress bar
    self.pbar.n = self.step
    self.pbar.last_print_n = self.step
    self.pbar.refresh()

    strategy.lr_callbacks.append(self.log_lr)