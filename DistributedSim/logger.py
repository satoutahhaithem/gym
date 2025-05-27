import wandb
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import os

from .sim_config import *
from .utils import extract_wandb_config, create_wandb_config


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


class WandbLogger(Logger):
  def __init__(self,
               model: nn.Module,
               max_steps: int,
               strategy=None,
               config=None,
               rank: int = 0,
               wandb_project: str = None,
               wandb_name: str = None,
               save_dir: str = "./checkpoints"):
    super().__init__(model, max_steps)
    
    self.config = config
    self.rank = rank
    self.wandb_project = wandb_project or (config.wandb_project if config else None)
    self.wandb_name = wandb_name or (config.wandb_name if config else None)
    self.save_dir = save_dir or (config.save_dir if config else "./checkpoints")

    # Create wandb configuration using the utility function
    wandb_config = create_wandb_config(
      model=model,
      strategy=strategy,
      config=config,
      extra_config={
        "max_steps": max_steps,
        "rank": rank,
      }
    )

    run_id_to_resume_with = None
    run_id_file_path = None

    # Only attempt to read/write a run_id file if a specific wandb_name is provided
    if self.wandb_name:
      experiment_dir = os.path.join(self.save_dir,
                                    self.wandb_project if self.wandb_project else 'default_project',
                                    self.wandb_name)
      os.makedirs(experiment_dir, exist_ok=True)
      run_id_file_path = os.path.join(experiment_dir, "wandb_run_id.txt")

      if os.path.exists(run_id_file_path):
        try:
          with open(run_id_file_path, "r") as f:
            run_id_to_resume_with = f.read().strip()
          if not run_id_to_resume_with: # Handle empty file
            run_id_to_resume_with = None
          else:
            print(f"Rank {self.rank}: Found saved run ID {run_id_to_resume_with} for run name {self.wandb_name}")
        except Exception as e:
          print(f"Rank {self.rank}: Could not read run ID from {run_id_file_path}: {e}. Will attempt to start new/resume by name.")
          run_id_to_resume_with = None
    
    init_kwargs = {
      "project": self.wandb_project,
      "name": self.wandb_name, # Can be None
      "config": wandb_config,
      "resume": "allow" # Allow resuming if possible, or create new
    }
    if run_id_to_resume_with:
      init_kwargs["id"] = run_id_to_resume_with

    wandb.init(**init_kwargs)

    self.wandb_name = wandb.run.name # Actual name of the run (could be generated)
    
    # Save/Update the run ID using the actual run name and ID, if wandb_name was configured.
    # This ensures that if config.wandb_name was set, we are managing the ID for that intended named run.
    if self.wandb_name and run_id_file_path: # run_id_file_path is only set if config.wandb_name was
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

    # Set current learning rate if available
    self.current_lr = 0.0
    try:
      if strategy and hasattr(strategy, 'optimizer_kwargs'):
        self.current_lr = strategy.optimizer_kwargs.get('lr', 0.0)
      elif config and hasattr(config, 'strategy_config'):
        strategy_config = config.strategy_config
        if hasattr(strategy_config, 'optimizer_kwargs') and strategy_config.optimizer_kwargs:
          self.current_lr = strategy_config.optimizer_kwargs.get('lr', 0.0)
    except:
      pass

  def log_config_update(self, config_dict: dict, prefix: str = ""):
    """
    Log additional configuration to wandb during training.
    
    Args:
      config_dict: Dictionary of configuration to log
      prefix: Optional prefix for the keys
    """
    if hasattr(self, 'wandb_name'):
      safe_config = extract_wandb_config(config_dict)
      if prefix:
        safe_config = {f"{prefix}_{k}": v for k, v in safe_config.items()}
      wandb.config.update(safe_config)
    