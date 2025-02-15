import wandb
from tqdm import tqdm
import numpy as np
import torch
from torch import nn

from .sim_config import *

class WandbLogger:
    def __init__(self, rank: int, device: torch.device, config: SimConfig, model: nn.Module, max_steps: int, project: str):
        self.device = device
        self.config = config
        self.project = project
        self.rank = rank
        self.step = 0

        assert self.rank == 0, "WandbLogger should only be initialized on rank 0"

        self.pbar = tqdm(total=max_steps)

        # Prepare the wandb config
        wandb_config = self.config.__dict__.copy()
        wandb_config.update({
            "architecture": "GPT",
            "model_parameters": model.get_num_params() / 1e6,
            "dataset": self.config.dataset_name,
        })

        print("MODEL PARAMS: ", model.get_num_params() / 1e6)

        # Remove unnecessary keys
        keys_to_remove = ['model_class', 'gpt_config', 'train_dataset', 'val_dataset']
        for key in keys_to_remove:
            if key in wandb_config:
                del wandb_config[key]

        wandb_config['gradient_config'] = self.config.gradient_config.__dict__

        wandb.init(project=self.config.wandb_project,
                   name=self.config.wandb_run_name, 
                   config=wandb_config)
        self.wandb_run_id = wandb.run.name

        self.current_lr = (self.config.gradient_config.optimizer_kwargs.get('lr', 0.0)
                           if self.config.gradient_config.optimizer_kwargs else 0.0)

    def log_pure(self, loss: float, name: str):
        data = {
            f"{name}_loss": loss,
            f"{name}_perplexity": float(np.exp(loss))
        }
        wandb.log(data, step=self.step)

    def log_train(self, loss: float):
        data = {
            "train_loss": loss,
            "train_perplexity": float(np.exp(loss)),
            "lr": self.current_lr
        }
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

    def log_dict(self, log_dict: dict):
        """
        Log an arbitrary dictionary of metrics.
        """
        self._distributed_log(log_dict)
