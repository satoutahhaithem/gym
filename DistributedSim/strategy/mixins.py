import torch.distributed as dist
from copy import deepcopy
import torch
from typing import Optional

from .communicate import *
from .optim import OptimSpec

class OuterOptMixin:
  """
  Mixin class containing shared outer optimization functionality for strategies
  that use a master model and outer optimizer (like DiLoCo).
  """
  
  def __init__(self, outer_optim_spec: Optional[OptimSpec] = None, **kwargs):
    super().__init__(**kwargs)
    
    if outer_optim_spec is not None:
      self.outer_optim_spec = outer_optim_spec
    else:
      self.outer_optim_spec = OptimSpec(
        torch.optim.SGD,
        lr=0.7,
        nesterov=True,
        momentum=0.9)

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