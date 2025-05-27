from dataclasses import dataclass, field
from typing import Optional, Callable, Type
import torch

from .strategy.strategy import *

class SimConfig:
    def __init__(self,
                 num_nodes: Optional[int] = None,
                 model_class: Optional[Type[torch.nn.Module]] = None,
                 model_kwargs: dict = {},
                 batch_size: int = 64,
                 strategy_class: Optional[Type[Strategy]] = None,
                 strategy_config: dict = {},
                 eval_interval: int = 10,
                 gpu_offset: int = 0,
                 checkpoint_interval: int = 100,
                 **kwargs):
        self.num_nodes = num_nodes

        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.batch_size = batch_size

        self.strategy_class = strategy_class
        self.strategy_config = strategy_config
        self.eval_interval = eval_interval
        self.gpu_offset = gpu_offset
        self.checkpoint_interval = checkpoint_interval

        # Allow additional kwargs to be set as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)