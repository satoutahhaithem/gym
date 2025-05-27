import torch
import torch.distributed as dist

from DistributedSim.train_node import TrainNode
from DistributedSim.strategy import Strategy

import os
from abc import ABC, abstractmethod
import copy

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
      
  def _get_split_dataset(self, dataset: torch.utils.data.Dataset, num_nodes: int):
    # TODO: Split dataset properly. Potentially we can have a subsampling dataset class.
    return copy.deepcopy(dataset)


  def fit(self,
          num_epochs: int,
          strategy: Strategy,
          num_nodes: int,
          device: str):
    self.device = device
    self.strategy = strategy
    self.num_nodes = num_nodes
    self.num_epochs = num_epochs

    print('pre spawn')
    torch.multiprocessing.spawn(self._fit, args=(), nprocs=num_nodes, join=True)

  def _fit(self, rank):
    self.rank = rank

    self._build_connection()

    self.train_dataset = self._get_split_dataset(self.train_dataset, rank)
    self.val_dataset = self._get_split_dataset(self.val_dataset, rank)

    self.model = copy.deepcopy(self.model).to(self.device)

    # TODO: Replace this with something native to strategy. Let the strategy reset it's own internal state.
    self.strategy = copy.deepcopy(self.strategy)
    self.strategy._init_node(self.model, self.rank, self.num_nodes)

    sim = TrainNode(
      self.model,
      self.train_dataset,
      self.val_dataset,
      self.strategy,
      self.device,
      self.rank,
      self.num_nodes
    )

    sim.train(num_epochs=self.num_epochs)

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

    print('a', self.device)

    if self.device == '' and torch.cuda.is_available():
        self.device = 'cuda'
    elif self.device == '' and torch.backends.mps.is_available():
        self.device = 'mps' 
    elif self.device == '':
        self.device = 'cpu'

    print('b', self.device)

    # initialize the process group
    if self.device == 'cuda':
        # If we haven't specified devices, use all devices.
        if not self.devices:
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