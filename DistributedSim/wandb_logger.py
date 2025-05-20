import wandb
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import os

from .sim_config import *

class WandbLogger:
    def __init__(self, rank: int, device: torch.device, config: SimConfig, model: nn.Module, max_steps: int):
        self.device = device
        self.config = config
        self.rank = rank
        self.step = 0

        assert self.rank == 0, "WandbLogger should only be initialized on rank 0"

        self.pbar = tqdm(total=max_steps, initial=0)

        if self.config.wandb_project is not None:
            # Prepare the wandb config
            wandb_config = self.config.__dict__.copy()
            wandb_config.update({
                "architecture": "GPT",
                "model_parameters": model.get_num_params() / 1e6,
                "dataset": self.config.dataset_name,
            })

            # Remove unnecessary keys
            keys_to_remove = ['model_class', 'model_config', 'train_dataset', 'val_dataset']
            for key in keys_to_remove:
                if key in wandb_config:
                    del wandb_config[key]

            if hasattr(self.config.gradient_config, '__dict__'):
                wandb_config['gradient_config'] = self.config.gradient_config.__dict__
            else:
                # Handle cases where gradient_config might not be a class with __dict__ (e.g. a simple dict)
                wandb_config['gradient_config'] = vars(self.config.gradient_config) if not isinstance(self.config.gradient_config, dict) else self.config.gradient_config

            run_id_to_resume_with = None
            run_id_file_path = None

            # Only attempt to read/write a run_id file if a specific wandb_name is provided
            if self.config.wandb_name:
                experiment_dir = os.path.join(self.config.save_dir,
                                              self.config.wandb_project, # wandb_project must be set if we are here
                                              self.config.wandb_name)
                os.makedirs(experiment_dir, exist_ok=True)
                run_id_file_path = os.path.join(experiment_dir, "wandb_run_id.txt")

                if os.path.exists(run_id_file_path):
                    try:
                        with open(run_id_file_path, "r") as f:
                            run_id_to_resume_with = f.read().strip()
                        if not run_id_to_resume_with: # Handle empty file
                            run_id_to_resume_with = None
                        else:
                            print(f"Rank {self.rank}: Found saved run ID {run_id_to_resume_with} for run name {self.config.wandb_name}")
                    except Exception as e:
                        print(f"Rank {self.rank}: Could not read run ID from {run_id_file_path}: {e}. Will attempt to start new/resume by name.")
                        run_id_to_resume_with = None
            
            init_kwargs = {
                "project": self.config.wandb_project,
                "name": self.config.wandb_name, # Can be None
                "config": wandb_config,
                "resume": "allow" # Allow resuming if possible, or create new
            }
            if run_id_to_resume_with:
                init_kwargs["id"] = run_id_to_resume_with

            wandb.init(**init_kwargs)

            self.wandb_name = wandb.run.name # Actual name of the run (could be generated)
            
            # Save/Update the run ID using the actual run name and ID, if wandb_name was configured.
            # This ensures that if config.wandb_name was set, we are managing the ID for that intended named run.
            if self.config.wandb_name and run_id_file_path: # run_id_file_path is only set if config.wandb_name was
                try:
                    with open(run_id_file_path, "w") as f:
                        f.write(wandb.run.id)
                    # print(f"Rank {self.rank}: Saved/Updated wandb run ID {wandb.run.id} to {run_id_file_path} for run {self.wandb_name}")
                except Exception as e:
                    print(f"Rank {self.rank}: Warning - Could not write run ID to {run_id_file_path}: {e}")
            
            # Set the logger's step based on wandb's step for the run
            self.step = wandb.run.step
            if wandb.run.resumed:
                print(f"Rank {self.rank}: Resumed wandb run '{self.wandb_name}' (ID: {wandb.run.id}). Current step is {self.step}.")
            else:
                print(f"Rank {self.rank}: Started new wandb run '{self.wandb_name}' (ID: {wandb.run.id}). Starting at step {self.step}.")

            # Update tqdm progress bar
            self.pbar.n = self.step
            self.pbar.last_print_n = self.step
            self.pbar.refresh()

            self.current_lr = (self.config.gradient_config.optimizer_kwargs.get('lr', 0.0)
                            if self.config.gradient_config.optimizer_kwargs else 0.0)

    def log(self, data: dict):
        if hasattr(self, 'wandb_name'):
            wandb.log(data, step=self.step)

    def log_pure(self, loss: float, name: str):
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