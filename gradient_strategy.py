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

    def zero_grad(self):
        self.optim.zero_grad()

class SimpleReduceGradient(GradientStrategy):
    def __init__(self, model, config):
        super().__init__(model, config)

        self.optim = config.optimizer_class(model.parameters(), **config.optimizer_kwargs)

    def step(self):
        # Default all_reduce, but doing it manually. 
        for name, param in self.model.named_parameters():
            ##  print(name, param, param.grad)
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= dist.get_world_size()

        self.optim.step()

## If we want to replace dist.all_reduce with dist.all_gather for consistency with DeMo.
class SimpleGatherGradient(GradientStrategy):
    def __init__(self, model, config):
        super().__init__(model, config)

        self.optim = config.optimizer_class(model.parameters(), **config.optimizer_kwargs)

    def step(self):
        world_size = dist.get_world_size()

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # Initialize a list to hold the gathered gradients for this parameter.
                gathered_gradients = [torch.zeros_like(param.grad) for _ in range(world_size)]

                # Use all_gather to collect gradients from all processes.
                dist.all_gather(gathered_gradients, param.grad)

                # Manually average the gathered gradients.
                avg_gradient = sum(gathered_gradients) / world_size

                # Update the gradient of the parameter to the averaged gradient.
                param.grad.copy_(avg_gradient)

        # Perform the optimization step.
        self.optim.step()



class DeMoGradient(GradientStrategy):
    def __init__(self, model, config):
        super().__init__(model, config)

        print('initialising DeMo engine')

        self.optim = DeMo(model.parameters(), **config.optimizer_kwargs)

    def step(self):
        # DeMo communicates gradients and then does optimizer step.
        self.optim.step()
