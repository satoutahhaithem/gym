import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group

from typing import Optional, Callable, Type
import os
from abc import ABC, abstractmethod

from .demo import *

class GradientConfig:
    def __init__(self, 
                 optimizer_class: Type[torch.optim.Optimizer] = None, 
                 optimizer_kwargs: dict = None,
                 lr_scheduler: Type[torch.optim.lr_scheduler._LRScheduler] = None,
                 lr_scheduler_kwargs: dict = None):
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs

class GradientStrategy:
    def __init__(self, rank, model, config, logger=None):
        self.rank = rank
        self.model = model
        self.config = config
        self.gradient_config = config.gradient_config

        if logger is not None:
            self.logger = logger

        self.nbytes = 0
        # Initialize scheduler as None; will be set after self.optim is defined in subclasses.
        self.scheduler = None

    @abstractmethod
    def step(self):
        self.nbytes = 0

        if self.scheduler is not None:
            self.scheduler.step()
            if self.rank == 0:
                self.logger.log_lr(self.scheduler.get_last_lr()[0])

    def zero_grad(self):
        self.optim.zero_grad()

    def all_gather(self, tensor_list, tensor, group=None, async_op=False):
        # Custom logic to save tensor size etc.
        nbytes = tensor.element_size() * tensor.nelement()
        self.nbytes += nbytes

        if self.config.num_nodes > 1:
            tensor_handle = dist.all_gather(tensor_list, tensor, group, async_op)
        else:
            tensor_list[0] = tensor
            tensor_handle = tensor_list[0]

        return tensor_handle

    def _setup_scheduler(self):
        if self.gradient_config.lr_scheduler is not None:
            lr_sched_kwargs = (self.gradient_config.lr_scheduler_kwargs 
                               if self.gradient_config.lr_scheduler_kwargs is not None else {})
            self.scheduler = self.gradient_config.lr_scheduler(self.optim, **lr_sched_kwargs)
        else:
            self.scheduler = None


class SimpleReduceGradient(GradientStrategy):
    def __init__(self, rank, model, config, logger=None):
        super().__init__(rank, model, config, logger)
        self.optim = self.gradient_config.optimizer_class(model.parameters(), 
                                                          **self.gradient_config.optimizer_kwargs)
        self._setup_scheduler()

    def step(self):
        # Default all_reduce, but doing it manually.
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= dist.get_world_size()

        self.optim.step()
        if self.scheduler is not None:
            self.scheduler.step()

class SimpleGatherGradient(GradientStrategy):
    def __init__(self, rank, model, config, logger=None):
        super().__init__(rank, model, config, logger)
        self.optim = self.gradient_config.optimizer_class(model.parameters(), 
                                                          **self.gradient_config.optimizer_kwargs)
        self._setup_scheduler()

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
    def __init__(self, rank, model, config, logger=None):
        super().__init__(rank, model, config, logger)
        print('initialising DeMo engine')
        self.optim = DeMo(model.parameters(), 
                          **self.gradient_config.optimizer_kwargs, 
                          custom_all_gather=super().all_gather)
        self._setup_scheduler()

    def step(self):
        # DeMo communicates gradients and then does optimizer step.
        self.optim.step()
        if self.scheduler is not None:
            self.scheduler.step()

        super().step()  # Print number of bytes communicated. This can be put in a different method tbh.
