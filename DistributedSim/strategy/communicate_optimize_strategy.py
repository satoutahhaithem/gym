import torch
from torch.nn import utils as nn_utils
from typing import List, Optional
from abc import ABC, abstractmethod

from .strategy import Strategy
from .optim import OptimSpec

class CommunicationModule(ABC):
  """Abstract base class for communication modules."""
  
  def __init__(self, strategy=None):
    self.strategy = strategy
  
  @abstractmethod
  def communicate(self, model, rank: int, num_nodes: int, local_step: int) -> None:
    """
    Perform communication for the given model.
    
    Args:
      model: The model to communicate
      rank: Current node rank
      num_nodes: Total number of nodes
      local_step: Current local step count
    """
    pass

class CommunicateOptimizeStrategy(Strategy):
  """
  Base strategy class for algorithms that separate local optimization 
  and communication into distinct phases.
  
  This strategy:
  1. Performs local optimization step
  2. Applies communication modules when the derived strategy decides
  """
  
  def __init__(self, 
               optim_spec: OptimSpec,
               communication_modules: List[CommunicationModule],
               max_norm: Optional[float] = None,
               **kwargs):
    super().__init__(**kwargs)
    
    self.optim_spec = optim_spec
    self.communication_modules = communication_modules
    self.max_norm = max_norm
    
    # Set strategy reference in communication modules that need it
    for comm_module in self.communication_modules:
      comm_module.strategy = self

  def step(self):
    # Gradient clipping if specified
    if self.max_norm:
      nn_utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm)

    # Local optimization step
    self.optim.step()

    # Communication phase - let derived strategies decide when
    self._communicate()

    super().step()

  def _communicate(self):
    """Apply all communication modules sequentially. Override in derived classes for custom timing."""
    for comm_module in self.communication_modules:
      comm_module.communicate(self.model, self.rank, self.num_nodes, self.local_step)

  def _init_node(self, model, rank, num_nodes):
    super()._init_node(model, rank, num_nodes)
    
    self.optim = self.optim_spec.build(model)
    self._setup_scheduler() 