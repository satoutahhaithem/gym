import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group

from typing import Optional, Callable, Type
import os
from abc import ABC, abstractmethod


class GradientStrategy:
    def __init__(self, model, config):
        # TODO: custom delay to communication speed
        self.model = model
        self.config = config

    @abstractmethod
    def communicate(self):
        pass

class SimpleReduceGradient(GradientStrategy):
    def __init__(self, model, config):
        super().__init__(model, config)

    def communicate(self):
        for param in self.model.parameters():
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= dist.get_world_size()

