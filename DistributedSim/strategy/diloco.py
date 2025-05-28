import torch.distributed as dist
from copy import deepcopy

from torch.nn import utils as nn_utils
import torch

from typing import Optional

from .communicate_optimize_strategy import CommunicateOptimizeStrategy, CommunicationModule
from .mixins import OuterOptMixin
from .optim import OptimSpec
from .communicate import *

class DiLoCoCommunicator(OuterOptMixin, CommunicationModule):
  """
  Communication module for master-worker setup (like DiLoCo).
  Inherits from OuterOptMixin to get master model functionality.
  """
  
  def __init__(self, 
               H: int=100, 
               outer_optim_spec: Optional[OptimSpec] = None, 
               **kwargs):
    super().__init__(outer_optim_spec=outer_optim_spec, **kwargs)

    self.H = H
  
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

class DiLoCoStrategy(CommunicateOptimizeStrategy):
  def __init__(self, 
               inner_optim_spec: Optional[OptimSpec] = None,
               outer_optim_spec: Optional[OptimSpec] = None,
               H: int = 100,
               **kwargs):
    self.H = H

    if inner_optim_spec is None:
      inner_optim_spec = OptimSpec(torch.optim.AdamW)
    
    # Create the master-worker communicator
    communicator = DiLoCoCommunicator(H=H, outer_optim_spec=outer_optim_spec)
    
    super().__init__(
      optim_spec=inner_optim_spec,
      communication_modules=[communicator],
      **kwargs
    )
    
    self.communicator = communicator

  def _communicate(self):
    """Apply communication modules at the specified frequency."""
    if self.local_step % self.H == 0 and self.local_step > 0:
      super()._communicate()

  def _init_node(self, model, rank, num_nodes):
    super()._init_node(model, rank, num_nodes)
    
    # Initialize master model for DiLoCo in the communicator
    self.communicator._init_master_model(model, rank)
    
    # Share references so the communicator can access the model
    self.communicator.model = self.model