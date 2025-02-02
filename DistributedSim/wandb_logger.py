import wandb
from tqdm import tqdm

from torch import nn

from .sim_config import *

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

