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
               inner_optim: Optional[OptimSpec] = None,
               outer_optim: Optional[OptimSpec] = None,
               p_sparta: float = 0.005,
               H: int = 100,
               **kwargs):
   
    # Create both communication modules
    index_selector = RandomIndexSelector(p_sparta)
    sparse_comm = SparseCommunicator(index_selector)
    diloco_comm = DiLoCoCommunicator(H=H, outer_optim=outer_optim)
    
    super().__init__(
      communication_modules=[sparse_comm, diloco_comm],  # Sparse comm happens every step
      inner_optim=inner_optim,
      **kwargs
    )
    
    self.index_selector = index_selector
