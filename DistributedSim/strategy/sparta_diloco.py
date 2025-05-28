import torch
from typing import Optional

from .communicate_optimize_strategy import CommunicateOptimizeStrategy
from .optim import OptimSpec
from .sparta import SparseCommunicator, RandomIndexSelector
from .diloco import DiLoCoCommunicator

class SPARTADiLoCoStrategy(CommunicateOptimizeStrategy):
  """
  Strategy that combines SPARTA's sparse communication with DiLoCo's master-worker optimization.
  
  This strategy:
  1. Performs local optimization 
  2. Applies sparse communication every step (SPARTA)
  3. Applies master-worker optimization every H steps (DiLoCo)
  """
  
  def __init__(self, 
               inner_optim_spec: Optional[OptimSpec] = None,
               outer_optim_spec: Optional[OptimSpec] = None,
               p_sparta: float = 0.005,
               H: int = 100,
               **kwargs):

    if inner_optim_spec is None:
      inner_optim_spec = OptimSpec(torch.optim.AdamW)
    
    # Create both communication modules
    index_selector = RandomIndexSelector(p_sparta)
    sparse_comm = SparseCommunicator(index_selector)
    diloco_comm = DiLoCoCommunicator(H=H, outer_optim_spec=outer_optim_spec)
    
    super().__init__(
      optim_spec=inner_optim_spec,
      communication_modules=[sparse_comm],  # Sparse comm happens every step
      **kwargs
    )
    
    self.diloco_comm = diloco_comm
    # self.diloco_comm.strategy = self
    self.diloco_H = H  # DiLoCo communication frequency
    self.index_selector = index_selector

  def _communicate(self):
    """Apply communication modules with different frequencies."""
    # SPARTA sparse communication every step
    super()._communicate()
    
    # DiLoCo master-worker communication every H steps
    if self.local_step % self.diloco_H == 0 and self.local_step > 0:
      self.diloco_comm.communicate(self.model, self.rank, self.num_nodes, self.local_step)

  def _init_node(self, model, rank, num_nodes):
    super()._init_node(model, rank, num_nodes)
    
    # Initialize master model for DiLoCo in the communicator
    self.diloco_comm._init_master_model(model, rank)
    
    # Share references so the communicator can access the model
    self.diloco_comm.model = self.model 