from dataclasses import dataclass, field
from typing import Optional, Callable, Type
import torch

from .gradient_strategy import *

class SimConfig:
    def __init__(self,
                 num_nodes: Optional[int] = None,
                 connection_callback: Optional[Callable[[], None]] = None,
                 model_class: Optional[Type[torch.nn.Module]] = None,
                 model_kwargs: dict = {},
                 train_dataset: Optional[torch.utils.data.Dataset] = None,
                 val_dataset: Optional[torch.utils.data.Dataset] = None,
                 batch_size: int = 64,
                 optimizer_class: Type[torch.optim.Optimizer] = torch.optim.SGD,
                 optimizer_kwargs: dict = {},
                 criterion_class: Optional[Type[torch.nn.Module]] = None,
                 criterion_kwargs: dict = {},
                 gradient_class: Optional[Type[GradientStrategy]] = None,
                 **kwargs):
        self.num_nodes = num_nodes
        self.connection_callback = connection_callback
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.criterion_class = criterion_class
        self.criterion_kwargs = criterion_kwargs
        self.gradient_class = gradient_class
        
        # Allow additional kwargs to be set as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
