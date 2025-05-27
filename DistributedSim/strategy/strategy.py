import torch
import torch.distributed as dist

from torch.optim.lr_scheduler import LambdaLR

from typing import Type
import math

import torch.nn.utils as nn_utils

from .communicate import *

class StrategyConfig:
    def __init__(self, 
                 optimizer_class: Type[torch.optim.Optimizer] = None, 
                 optimizer_kwargs: dict = None,
                 lr_scheduler: Type[torch.optim.lr_scheduler._LRScheduler] = None,
                 lr_scheduler_kwargs: dict = None,
                 max_local_steps: int = None,
                 max_norm: float = None,
                 **kwargs):
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.max_local_steps = max_local_steps
        self.max_norm = max_norm

        # Allow additional kwargs to be set as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

class Strategy:
    def __init__(self):
        # if logger is not None:
        #     self.logger = logger

        # Initialize scheduler as None; will be set after self.optim is defined in subclasses.
        self.scheduler = None

    def _init_node(self, model, rank, num_nodes):
        self.model = model
        self.rank = rank
        self.num_nodes = num_nodes

        self.local_step = 0

        # TODO: Ensure we clean state.

    # @abstractmethod
    def step(self):
        self.nbytes = 0

        if self.scheduler is not None:
            self.scheduler.step()
            if self.rank == 0:
                self.logger.log_lr(self.scheduler.get_last_lr()[0])

        self.local_step += 1

    def zero_grad(self):
        self.optim.zero_grad()

    def _setup_scheduler(self):
        return # TODO

        def lr_lambda(current_step):
            if current_step < self.strategy_config.warmup_steps:
                return float(current_step) / float(max(self.strategy_config.warmup_steps, 1))
            elif self.strategy_config.cosine_anneal:
                min_lr_factor = 0.1
                progress = (current_step - self.strategy_config.warmup_steps) / float(
                    max(1, self.strategy_config.max_local_steps - self.strategy_config.warmup_steps)
                )
                cosine_term = 0.5 * (1.0 + math.cos(math.pi * progress))
                return (1 - min_lr_factor) * cosine_term + min_lr_factor
            else:
                return 1.0
            
        if self.strategy_config.lr_scheduler == 'lambda_cosine':
            self.scheduler = LambdaLR(self.optim, lr_lambda)
        elif self.strategy_config.lr_scheduler is not None:
            lr_sched_kwargs = (self.strategy_config.lr_scheduler_kwargs 
                               if self.strategy_config.lr_scheduler_kwargs is not None else {})
            self.scheduler = self.strategy_config.lr_scheduler(self.optim, **lr_sched_kwargs)
        else:
            self.scheduler = None


class SimpleReduceStrategy(Strategy):
    def __init__(self, rank, model, config, logger=None):
        super().__init__(rank, model, config, logger)
        self.optim = self.strategy_config.optimizer_class(model.parameters(), 
                                                          **self.strategy_config.optimizer_kwargs)
        self._setup_scheduler()

    def step(self):
        if self.num_nodes > 1 or True:
            for param in self.model.parameters():
                if param.grad is not None:
                    all_reduce(param.grad)
                    param.grad.div_(self.num_nodes)

            if self.strategy_config.max_norm:
                nn_utils.clip_grad_norm_(self.model.parameters(), max_norm=self.strategy_config.max_norm)

        self.optim.step()

        super().step()