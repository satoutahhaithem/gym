import torch.distributed as dist
from copy import deepcopy

from torch.nn import utils as nn_utils
import torch

from typing import Optional

from .communicate_optimize_strategy import CommunicateOptimizeStrategy, CommunicationModule
from .optim import OptimSpec
from .communicate import *

class DiLoCoCommunicator(CommunicationModule):
  """
  Communication module for master-worker setup (like DiLoCo).
  """
  
  def __init__(self, 
               H: int=100, 
               outer_optim: Optional[OptimSpec] = None, 
               **kwargs):
    super().__init__(**kwargs)

    if outer_optim is not None:
      self.outer_optim_spec = outer_optim
    else:
      self.outer_optim_spec = OptimSpec(
        torch.optim.SGD,
        lr=0.7,
        nesterov=True,
        momentum=0.9)

    self.H = H

  def _average_models(self) -> None:
    """Average model parameters across all nodes."""
    for param in self.model.parameters():
      all_reduce(param.data, op=dist.ReduceOp.SUM)
      param.data /= self.num_nodes

  def _broadcast_model_params(self) -> None:
    """Broadcast model parameters from rank 0 to all other nodes."""
    for param in self.model.parameters():
      broadcast(param.data, src=0)

  def _set_master_grad(self) -> None:
    """Set gradients on master model based on difference between master and worker models."""
    for name, param in self.master_model.named_parameters():
      param.grad = param.data - self.model.state_dict()[name].data.to('cpu')

  def _synchronize_master_model(self) -> None:
    """Synchronize worker model with master model parameters."""
    for name, param in self.model.named_parameters():
      param.data = self.master_model.state_dict()[name].data.to(param.device)

  def _init_master_model(self, model, rank):
    """Initialize master model and outer optimizer (typically only on rank 0)."""
    if rank == 0:
      self.master_model = deepcopy(model).to("cpu")
      for param in self.master_model.parameters():
        param.requires_grad = True
      self.outer_optimizer = self.outer_optim_spec.build(self.master_model)  

  def communicate(self, model, rank: int, num_nodes: int, local_step: int) -> None:
    """Perform master-worker communication."""
    if num_nodes > 1 and local_step % self.H == 0:
      # First average all models
      for param in model.parameters():
        all_reduce(param.data, op=dist.ReduceOp.SUM)
        param.data /= num_nodes

      # Master does outer optimization step
      if rank == 0:
        # This assumes the strategy has master model functionality
        if hasattr(self, 'outer_optimizer') and hasattr(self, 'master_model'):
          self.outer_optimizer.zero_grad()
          self._set_master_grad()
          self.outer_optimizer.step()
          self._synchronize_master_model()

      # Broadcast updated parameters
      for param in model.parameters():
        broadcast(param.data, src=0)

  def _init_node(self, model, rank, num_nodes):
    self._init_master_model(model, rank)

    self.model = model


class DiLoCoStrategy(CommunicateOptimizeStrategy):
  def __init__(self, 
               inner_optim: Optional[OptimSpec] = None,
               outer_optim: Optional[OptimSpec] = None,
               H: int = 100,
               **kwargs):
    self.H = H

    if inner_optim is None:
      self.inner_optim_spec = OptimSpec(torch.optim.AdamW)
    else:
      self.inner_optim_spec = inner_optim

    # Create the master-worker communicator
    communicator = DiLoCoCommunicator(H=H, outer_optim=outer_optim)
    
    super().__init__(
      communication_modules=[communicator],
      inner_optim=inner_optim,
      **kwargs
    )
    
    self.communicator = communicator

  # def _init_node(self, model, rank, num_nodes):
  #   super()._init_node(model, rank, num_nodes)
    
  #   # Share references so the communicator can access the model
  #   self.communicator.model = self.model