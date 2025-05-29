import torch
import torch.distributed as dist
import numpy as np

from DistributedSim.train_node import TrainNode
from DistributedSim.strategy import Strategy

import os
from abc import ABC, abstractmethod
import copy

from typing import Tuple

# def print_dataset_size(dataset: torch.utils.data.Dataset):
#   from pympler import asizeof
#   print(f"Dataset size: {asizeof.asizeof(dataset)}")

def print_dataset_size(dataset: torch.utils.data.Dataset):
  import pickle, sys, io

  buffer = io.BytesIO()
  pickle.dump(dataset, buffer, protocol=pickle.HIGHEST_PROTOCOL)
  print(f"Dataset size: {buffer.tell() // 1024 // 1024} MB")


class Trainer:
  '''
  Trainer is used to train a model.
  '''
  def __init__(self, 
              model: torch.nn.Module,
              train_dataset: torch.utils.data.Dataset,
              val_dataset: torch.utils.data.Dataset,
              lazily_load_dataset: bool = False):
    self.model = model
    self.train_dataset = train_dataset
    self.val_dataset = val_dataset
    self.lazily_load_dataset = lazily_load_dataset

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
          val_size: int = 64,
          eval_interval: int = 100,
          autocast: bool = False,
          checkpoint_interval: int = 100,
          **kwargs):
    self.num_epochs = num_epochs
    self.max_steps = max_steps
    self.strategy = strategy
    self.num_nodes = num_nodes
    self.device = device
    self.devices = devices

    self.batch_size = batch_size
    self.minibatch_size = minibatch_size
    self.val_size = val_size
    self.eval_interval = eval_interval
    self.autocast = autocast
    self.checkpoint_interval = checkpoint_interval

    self.kwargs = kwargs

    # Set random seeds
    seed = kwargs.get('seed', 42)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    torch.multiprocessing.spawn(self._fit, args=(), nprocs=num_nodes, join=True)

  def _fit(self, rank):
    self.rank = rank

    self._build_connection()

    # print_dataset_size(self.train_dataset)

    self.model = copy.deepcopy(self.model).to(self.device)

    self.strategy = copy.deepcopy(self.strategy)
    self.strategy._init_node(self.model, self.rank, self.num_nodes)

    # self.sampler = torch.utils.data.DistributedSampler(self.train_dataset, num_replicas=self.num_nodes, rank=self.rank, shuffle=True)
    self.sampler = torch.utils.data.DistributedSampler(self.train_dataset, num_replicas=self.num_nodes, rank=self.rank, shuffle=False)

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

    if self.device == None and torch.cuda.is_available():
        self.device = 'cuda'
    elif self.device == None and torch.backends.mps.is_available():
        self.device = 'mps' 
    elif self.device == None:
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
        raise ValueError(f"Invalid device type: {self.config.device_type}")

    print(f"Rank {self.rank} using device {self.device}")