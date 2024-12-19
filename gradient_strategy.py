import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group

from typing import Optional, Callable, Type
import os
from abc import ABC, abstractmethod

from demo import *


class GradientStrategy:
    def __init__(self, model, config):
        # TODO: custom delay to communication speed
        self.model = model
        self.config = config

    @abstractmethod
    def step(self):
        pass

class SimpleReduceGradient(GradientStrategy):
    def __init__(self, model, config):
        super().__init__(model, config)

        self.optim = config.optimizer_class(model.parameters(), **config.optimizer_kwargs)

    def step(self):
        for name, param in self.model.named_parameters():
            ##  print(name, param, param.grad)
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= dist.get_world_size()

        self.optim.step()

        

class DeMoGradient(GradientStrategy):
    def __init__(self, model, config):
        super().__init__(model, config)

        print('initialising DeMo engine')

        self.demo = DeMo(model.parameters(), **config.optimizer_kwargs)

    def step(self):
        # DeMo communicates gradients and then does optimizer step.
        self.demo.step()
