import wandb
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import os

from .sim_config import *

class Logger:
  def __init__(self,
               model: nn.Module,
               max_steps: int):
    self.model = model
    self.max_steps = max_steps

    self.pbar = tqdm(total=self.max_steps, initial=0)

    print(f'Logger initialized.')
    ## TODO: More general get_num_params method
    print(f'Model parameter count: {self.model.get_num_params() / 1e6}M')

    self.local_step = 0
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
      wandb.log(data, step=self.local_step)

  def log_train(self, loss: float):
    if hasattr(self, 'wandb_name'):
      data = {
        "train_loss": loss,
        "train_perplexity": float(np.exp(loss)),
      }
      if self.current_lr:
        data["lr"] = self.current_lr

      wandb.log(data, step=self.local_step)

    self.pbar.update(1)
    self.pbar.set_postfix({
      "train_loss": f"{loss:.4f}",
      "lr": f"{self.current_lr:.6f}",
    })

  def increment_step(self):
    self.local_step += 1

  def log_lr(self, lr: float):
    self.current_lr = lr