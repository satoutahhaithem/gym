import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group

from typing import Optional, Callable, Type
import os
from abc import ABC, abstractmethod

from .demo import *

class GradientConfig:
    def __init__(self, optimizer_class: Type[torch.optim.Optimizer]=None, optimizer_kwargs: dict=None):
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs

class GradientStrategy:
    def __init__(self, model, config, logger=None):
        self.model = model
        self.config = config
        self.gradient_config = config.gradient_config

        if logger is not None:
            self.logger = logger

        self.nbytes = 0

    @abstractmethod
    def step(self):
        self.nbytes = 0

    def zero_grad(self):
        self.optim.zero_grad()

    def all_gather(self, tensor_list, tensor, group=None, async_op=False):
        ## Custom logic to save tensor size etc.
        nbytes = tensor.element_size() * tensor.nelement()
        self.nbytes += nbytes

        if self.config.num_nodes > 1:
            tensor_handle = dist.all_gather(tensor_list, tensor, group, async_op)
        else:
            tensor_list[0] = tensor
            tensor_handle = tensor_list[0]

        return tensor_handle

class SimpleReduceGradient(GradientStrategy):
    def __init__(self, model, config, logger=None):
        super().__init__(model, config, logger)


        self.optim = self.gradient_config.optimizer_class(model.parameters(), 
                                                          **self.gradient_config.optimizer_kwargs)

    def step(self):
        # Default all_reduce, but doing it manually. 
        for name, param in self.model.named_parameters():
            ##  print(name, param, param.grad)
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= dist.get_world_size()

        self.optim.step()

## If we want to replace dist.all_reduce with dist.all_gather for consistency with DeMo.
class SimpleGatherGradient(GradientStrategy):
    def __init__(self, model, config, logger=None):
        super().__init__(model, config, logger)

        self.optim = self.gradient_config.optimizer_class(model.parameters(), 
                                                          **self.gradient_config.optimizer_kwargs)

    def step(self):
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # Initialize a list to hold the gathered gradients for this parameter.
                gathered_gradients = [torch.zeros_like(param.grad) for _ in range(self.config.num_nodes)]

                # Use all_gather to collect gradients from all processes.
                super().all_gather(gathered_gradients, param.grad)

                # Manually average the gathered gradients.
                avg_gradient = sum(gathered_gradients) / self.config.num_nodes

                # Update the gradient of the parameter to the averaged gradient.
                param.grad.copy_(avg_gradient)

        # Perform the optimization step.
        self.optim.step()

        super().step()



class DeMoGradient(GradientStrategy):
    def __init__(self, model, config, logger=None):
        super().__init__(model, config, logger)

        print('initialising DeMo engine')

        self.optim = DeMo(model.parameters(), 
                          **self.gradient_config.optimizer_kwargs, 
                          custom_all_gather=super().all_gather)

    def step(self):
        # DeMo communicates gradients and then does optimizer step.
        self.optim.step()

        super().step() # Print number of bytes communicated. This can be put in a different method tbh.
