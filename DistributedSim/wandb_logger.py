import wandb
from tqdm import tqdm
import numpy as np

from torch import nn

from .sim_config import *

# class EvalStats:
#     def __init__(self, loss: float=None, lr: float=None, train: bool=True):
#         self.loss = loss
#         # self.lr = lr
#         self.perplexity = np.exp(loss)

#         self.train = train

#     def wandb_dict(self):
#         if self.train:
#             return {
#                 "train_loss": self.loss,
#                 "train_perplexity": self.perplexity,
#                 # "lr": self.lr
#             }
#         else:
#             return {
#                 "val_loss": self.loss,
#                 "val_perplexity": self.perplexity,
#             }

class WandbLogger:
    def __init__(self, config: SimConfig, model: nn.Module, max_steps: int, project: str):
        self.config = config
        self.project = project

        self.pbar = tqdm(total=max_steps)

        self.step = 0

        wandb_config = self.config.__dict__.copy()
        wandb_config.update({
            "architecture": "GPT",
            "model_parameters": model.get_num_params() / 1e6,
        })

        del wandb_config['model_class']
        del wandb_config['gpt_config']
        del wandb_config['train_dataset']
        del wandb_config['val_dataset']
        
        wandb.init(
            project=self.config.wandb_project,
            config=wandb_config
        )

    def log_train(self, loss: float):
        wandb.log({
            "train_loss": loss,
            "train_perplexity": np.exp(loss)
        }, step=self.step)

        self.pbar.update(1)
        self.pbar.set_postfix(
            {
                "train_loss": f"{loss:.4f}",
            }
        )

        self.step += 1

    def log_val(self, loss: float):
        wandb.log({
            "val_loss": loss,
            "val_perplexity": np.exp(loss)
        }, step=self.step)
    
    def log_dict(self, dict: dict):
        '''
        Log a dictionary of metrics - for use from GradientStrategy.
        '''
        wandb.log(dict, step=self.step)
