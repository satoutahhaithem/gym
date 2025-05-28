import torch
import torch.distributed as dist

from DistributedSim.train_node import TrainNode
from DistributedSim.strategy import Strategy
from DistributedSim.partial_dataset import PartialDataset

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
      
  def _get_split_dataset(self, dataset: torch.utils.data.Dataset, rank: int, num_nodes: int):
    # TODO: Split dataset properly. Potentially we can have a subsampling dataset class.
    return PartialDataset(dataset, rank, num_nodes)


  def fit(self,
          num_epochs: int,
          strategy: Strategy,
          num_nodes: int,
          device: str,
          batch_size: int = 16,
          minibatch_size: int = 16,
          val_size: int = 64,
          eval_interval: int = 100,
          autocast: bool = False,
          checkpoint_interval: int = 100,
          **kwargs):
    self.device = device
    self.strategy = strategy
    self.num_nodes = num_nodes
    self.num_epochs = num_epochs

    self.batch_size = batch_size
    self.minibatch_size = minibatch_size
    self.val_size = val_size
    self.eval_interval = eval_interval
    self.autocast = autocast
    self.checkpoint_interval = checkpoint_interval

    self.kwargs = kwargs

    torch.multiprocessing.spawn(self._fit, args=(), nprocs=num_nodes, join=True)

  def _fit(self, rank):
    self.rank = rank

    self._build_connection()

    self.train_dataset = self._get_split_dataset(self.train_dataset, rank, self.num_nodes)
    self.val_dataset = self.val_dataset

    self.model = copy.deepcopy(self.model).to(self.device)

    self.strategy = copy.deepcopy(self.strategy)
    self.strategy._init_node(self.model, self.rank, self.num_nodes)

    sim = TrainNode(
      self.model,
      self.train_dataset,
      self.val_dataset,
      self.strategy,
      self.device,
      self.rank,
      self.num_nodes,
      num_epochs=self.num_epochs,
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

    if self.device == '' and torch.cuda.is_available():
        self.device = 'cuda'
    elif self.device == '' and torch.backends.mps.is_available():
        self.device = 'mps' 
    elif self.device == '':
        self.device = 'cpu'

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