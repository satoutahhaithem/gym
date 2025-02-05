import wandb
from tqdm import tqdm
import numpy as np

from torch import nn

from .sim_config import *

class WandbLogger:
    def __init__(self, rank: int, device: torch.device, config: SimConfig, model: nn.Module, max_steps: int, project: str):
        self.device = device
        self.config = config
        self.project = project
        self.rank = rank

        if self.rank == 0:
            self.pbar = tqdm(total=max_steps)
        else:
            self.pbar = None

        self.step = 0

        # Prepare the wandb config
        wandb_config = self.config.__dict__.copy()
        wandb_config.update({
            "architecture": "GPT",
            "model_parameters": model.get_num_params() / 1e6,
        })

        # Remove unnecessary keys
        keys_to_remove = ['model_class', 'gpt_config', 'train_dataset', 'val_dataset']
        for key in keys_to_remove:
            del wandb_config[key]

        wandb_config['gradient_config'] = self.config.gradient_config.__dict__

        # Handle wandb initialization across ranks
        if self.rank == 0:
            wandb.init(project=self.config.wandb_project,
                       name=config.wandb_run_name, 
                       config=wandb_config)
            run_id = wandb.run.id
            run_name = wandb.run.name
            # Broadcast run_id and run_name to other ranks
            self._broadcast_run_info(run_id, run_name)
        else:
            run_id, run_name = self._receive_run_info()
            # print(run_id, run_name)
            wandb.init(project=self.config.wandb_project, config=wandb_config,
                       id=run_id.strip(), name=run_name.strip(), resume="allow")

        self.wandb_run_id = wandb.run.name

        self.current_lr = self.config.gradient_config.optimizer_kwargs.get('lr', None) if \
            self.config.gradient_config.optimizer_kwargs else 0.0

    def log_pure(self, loss: float, name: str):
        wandb.log({
            f"{name}_loss": loss,
            f"{name}_perplexity": np.exp(loss)
        }, step=self.step)
        # print(f'logged {name} loss: {loss}')

    def log_train(self, loss: float):
        self.log_pure(loss, "train")

        if self.rank == 0:
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

    def log_lr(self, lr: float):
        if self.rank == 0:
            self.current_lr = lr

    def log_dict(self, dict: dict):
        '''
        Log a dictionary of metrics - for use from GradientStrategy.
        '''
        wandb.log(dict, step=self.step)

    def _broadcast_run_info(self, run_id: str, run_name: str):
        """Broadcast run ID and name from rank 0 to others."""
        max_length = 256
        run_id_encoded = run_id.ljust(max_length).encode('utf-8')
        run_name_encoded = run_name.ljust(max_length).encode('utf-8')
        
        run_id_tensor = torch.tensor(list(run_id_encoded), dtype=torch.uint8).to(self.device)
        run_name_tensor = torch.tensor(list(run_name_encoded), dtype=torch.uint8).to(self.device)
        
        torch.distributed.broadcast(run_id_tensor, src=0)
        torch.distributed.broadcast(run_name_tensor, src=0)

    def _receive_run_info(self) -> tuple:
        """Receive run ID and name from rank 0."""
        max_length = 256
        run_id_tensor = torch.zeros(max_length, dtype=torch.uint8).to(self.device)
        run_name_tensor = torch.zeros(max_length, dtype=torch.uint8).to(self.device)
        
        torch.distributed.broadcast(run_id_tensor, src=0)
        torch.distributed.broadcast(run_name_tensor, src=0)
        
        run_id = run_id_tensor.cpu().numpy().tobytes().decode('utf-8').strip('\x00')
        run_name = run_name_tensor.cpu().numpy().tobytes().decode('utf-8').strip('\x00')
        return run_id, run_name