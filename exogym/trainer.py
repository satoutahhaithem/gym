import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np

from exogym.train_node import TrainNode
from exogym.strategy import Strategy

import os
from abc import ABC, abstractmethod
import copy
from dataclasses import dataclass
from typing import Tuple, Optional, List, Any, Dict

# def print_dataset_size(dataset: torch.utils.data.Dataset):
#   from pympler import asizeof
#   print(f"Dataset size: {asizeof.asizeof(dataset)}")

def print_dataset_size(dataset: torch.utils.data.Dataset):
  import pickle, sys, io

  buffer = io.BytesIO()
  pickle.dump(dataset, buffer, protocol=pickle.HIGHEST_PROTOCOL)
  print(f"Dataset size: {buffer.tell() // 1024 // 1024} MB")


@dataclass
class TrainingConfig:
  """Configuration class that holds all training parameters for serialization."""
  model: torch.nn.Module
  train_dataset: torch.utils.data.Dataset
  val_dataset: torch.utils.data.Dataset
  strategy: Strategy
  num_epochs: int
  num_nodes: int
  max_steps: Optional[int] = None
  device: Optional[str] = None
  devices: Optional[List[int]] = None
  batch_size: int = 16
  minibatch_size: int = 16
  shuffle: bool = True
  val_size: int = 64
  eval_interval: int = 100
  autocast: bool = False
  checkpoint_interval: int = 100
  trainer_class: type = None
  kwargs: Dict[str, Any] = None

  def __post_init__(self):
    if self.kwargs is None:
      self.kwargs = {}


def _worker(rank: int, config: TrainingConfig):
  """
  Entry point executed in every child process.
  This function is importable as exogym.trainer._worker, making it notebook-safe.
  """
  # Create trainer instance in the worker process
  trainer = config.trainer_class(
    model=config.model,
    train_dataset=config.train_dataset,
    val_dataset=config.val_dataset,
  )
  
  # Set all the configuration parameters
  trainer.num_epochs = config.num_epochs
  trainer.max_steps = config.max_steps
  trainer.strategy = config.strategy
  trainer.num_nodes = config.num_nodes
  trainer.device = config.device
  trainer.devices = config.devices
  trainer.batch_size = config.batch_size
  trainer.minibatch_size = config.minibatch_size
  trainer.shuffle = config.shuffle
  trainer.val_size = config.val_size
  trainer.eval_interval = config.eval_interval
  trainer.autocast = config.autocast
  trainer.checkpoint_interval = config.checkpoint_interval
  trainer.kwargs = config.kwargs
  
  # Run the training process
  trainer._fit_process(rank)


def launch(config: TrainingConfig, num_nodes: int = None):
  """
  Spawn `num_nodes` processes using safe 'spawn' method.
  This is the only place where torch.multiprocessing.spawn is called.
  """
  if num_nodes is not None:
    config.num_nodes = num_nodes
    
  # Set random seeds before spawning
  seed = config.kwargs.get('seed', 42)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  np.random.seed(seed)

  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True

  mp.spawn(
    _worker,
    args=(config,),
    nprocs=config.num_nodes,
    start_method="spawn",
    join=True,
  )


class Trainer:
  '''
  Trainer is used to train a model.
  '''
  def __init__(self, 
              model: torch.nn.Module,
              train_dataset: torch.utils.data.Dataset,
              val_dataset: torch.utils.data.Dataset):
    self.model = model
    self.train_dataset = train_dataset
    self.val_dataset = val_dataset

    # print_dataset_size(self.train_dataset)
      
  def fit(self,
          num_epochs: int,
          strategy: Strategy,
          num_nodes: int,
          max_steps: int = None,
          device: str = None,
          devices: list[int] = None,
          batch_size: int = 16,
          minibatch_size: int = 16,
          shuffle: bool = True,
          val_size: int = 64,
          eval_interval: int = 100,
          autocast: bool = False,
          checkpoint_interval: int = 100,
          **kwargs):
    """
    Train the model. For single process training (num_nodes=1), runs directly.
    For multi-process training, delegates to launch_ddp for notebook safety.
    """
    # Store parameters
    self.num_epochs = num_epochs
    self.max_steps = max_steps
    self.strategy = strategy
    self.num_nodes = num_nodes
    self.device = device
    self.devices = devices
    self.batch_size = batch_size
    self.minibatch_size = minibatch_size
    self.shuffle = shuffle
    self.val_size = val_size
    self.eval_interval = eval_interval
    self.autocast = autocast
    self.checkpoint_interval = checkpoint_interval
    self.kwargs = kwargs

    if num_nodes == 1:
      # Single process mode - run directly for debugging
      # Set random seeds
      seed = kwargs.get('seed', 42)
      torch.manual_seed(seed)
      torch.cuda.manual_seed(seed)
      np.random.seed(seed)

      torch.backends.cuda.matmul.allow_tf32 = True
      torch.backends.cudnn.allow_tf32 = True
      
      self._fit_process(rank=0)
    else:
      # Multi-process mode - use safe launcher
      config = TrainingConfig(
        model=self.model,
        train_dataset=self.train_dataset,
        val_dataset=self.val_dataset,
        strategy=strategy,
        num_epochs=num_epochs,
        num_nodes=num_nodes,
        max_steps=max_steps,
        device=device,
        devices=devices,
        batch_size=batch_size,
        minibatch_size=minibatch_size,
        shuffle=shuffle,
        val_size=val_size,
        eval_interval=eval_interval,
        autocast=autocast,
        checkpoint_interval=checkpoint_interval,
        trainer_class=self.__class__,
        kwargs=kwargs
      )
      launch(config)

  def _fit_process(self, rank):
    """
    The core training logic that runs in each process.
    Renamed from _fit and removed the spawn call.
    """
    self.rank = rank

    self._build_connection()

    # print_dataset_size(self.train_dataset)

    self.model = copy.deepcopy(self.model).to(self.device)

    self.strategy = copy.deepcopy(self.strategy)
    self.strategy._init_node(self.model, self.rank, self.num_nodes)

    self.sampler = torch.utils.data.DistributedSampler(self.train_dataset, num_replicas=self.num_nodes, rank=self.rank, shuffle=self.shuffle)

    sim = TrainNode(
      self.model,
      self.train_dataset,
      self.sampler,
      self.val_dataset,
      self.strategy,
      self.device,
      self.rank,
      self.num_nodes,
      num_epochs=self.num_epochs,
      max_steps=self.max_steps,
      batch_size=self.batch_size,
      minibatch_size=self.minibatch_size,
      val_size=self.val_size,
      eval_interval=self.eval_interval,
      checkpoint_interval=self.checkpoint_interval,
      autocast=self.autocast,
      **self.kwargs
    )

    sim.train()

    self._process_cleanup()

  @abstractmethod
  def _build_connection(self):
    raise NotImplementedError

  def _process_cleanup(self):
    dist.destroy_process_group()


class LocalTrainer(Trainer):
  def _build_connection(self):
    '''
    This is the default callback for setting up pytorch distributed connections.
    All ranks are assumed to be on the same machine, and device is defaulted to cpu.
    '''
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(12355 + (10 if self.device == 'cpu' else 0))

    if self.device == '' or self.device == None:
      if torch.cuda.is_available():
        self.device = 'cuda'
      elif torch.backends.mps.is_available():
        self.device = 'mps' 
      else:
          self.device = 'cpu'

    # initialize the process group
    if self.device == 'cuda':
        # If we haven't specified devices, use all devices.
        if self.devices is None:
            self.devices = range(torch.cuda.device_count())

        dist.init_process_group("nccl" if len(self.devices) == self.num_nodes else "gloo", 
                                rank=self.rank, 
                                world_size=self.num_nodes)
        self.device = torch.device(f"cuda:{self.devices[self.rank % len(self.devices)]}")
        torch.cuda.set_device(self.device)
    elif self.device == 'cpu':
        dist.init_process_group("gloo", 
                                rank=self.rank, 
                                world_size=self.num_nodes)
        self.device = torch.device("cpu")
    elif self.device == 'mps':
        dist.init_process_group("gloo", 
                                rank=self.rank, 
                                world_size=self.num_nodes)
        self.device = torch.device("mps")
    else:
        raise ValueError(f"Invalid device type: {self.device}")

    print(f"Rank {self.rank} using device {self.device}")


# Script entry-point for CLI usage
if __name__ == "__main__":
  # Example usage - you'll need to adapt this to your actual configuration
  # from exogym.config import default_config
  # launch_ddp(default_config, num_nodes=4)
  print("Use launch_ddp() function or Trainer.fit() for training")