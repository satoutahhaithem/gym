import wandb
from tqdm import tqdm
import numpy as np

from torch import nn

from .sim_config import *

class WandbLogger:
    def __init__(self, 
                 config: SimConfig, 
                 max_steps: int, 
                 project: str):
        self.config = config
        self.project = project

        self.pbar = tqdm(total=max_steps)

        self.step = 0

        # Prepare the wandb config
        wandb_config = self.config.__dict__.copy()
        wandb_config.update({
            "architecture": "GPT",
            # "model_parameters": model.get_num_params() / 1e6,
        })

        # Remove unnecessary keys
        keys_to_remove = ['model_class', 'gpt_config', 'train_dataset', 'val_dataset']
        for key in keys_to_remove:
            del wandb_config[key]

        wandb_config['gradient_config'] = self.config.gradient_config.__dict__

        # Handle wandb initialization across ranks
        wandb.init(project=self.config.wandb_project, config=wandb_config)

        self.current_lr = self.config.gradient_config.optimizer_kwargs.get('lr', None) if \
            self.config.gradient_config.optimizer_kwargs else 0.0

        self.wandb_run_id = wandb.run.name

    def log_pure(self, loss: float, name: str):
        wandb.log({
            f"{name}_loss": loss,
            f"{name}_perplexity": np.exp(loss)
        }, step=self.step)
        # print(f'logged {name} loss: {loss}')

    def log_train(self, loss: float, rank: int):
        self.log_pure(loss, "train")

        if rank == 0:
            self.pbar.update(1)
            self.pbar.set_postfix(
                {
                    "train_loss": f"{loss:.4f}",
                    "lr": f"{self.current_lr:.6f}",
                }
            )

            wandb.log({
                "lr":self.current_lr
            }, step=self.step)

    def increment_step(self):
        self.step += 1

    def log_lr(self, lr: float, rank: int):
        if rank == 0:
            self.current_lr = lr

    def log_dict(self, dict: dict):
        '''
        Log a dictionary of metrics - for use from GradientStrategy.
        '''
        wandb.log(dict, step=self.step)