import torch.distributed as dist
from copy import deepcopy

from torch.nn import utils as nn_utils
import torch

from typing import Optional

from .strategy import Strategy
from .optim import OptimSpec
from .communicate import *

class DiLoCoStrategy(Strategy):
    def __init__(self, 
                inner_optim_spec: Optional[OptimSpec] = None,
                outer_optim_spec: Optional[OptimSpec] = None,
                diloco_interval: int = 100,
                **kwargs):

        super().__init__(**kwargs)

        if inner_optim_spec is not None:
            self.inner_optim_spec = inner_optim_spec
        else:
            self.inner_optim_spec = OptimSpec(
                torch.optim.AdamW
            )

        if outer_optim_spec is not None:
            self.outer_optim_spec = outer_optim_spec
        else:
            self.outer_optim_spec = OptimSpec(
                torch.optim.SGD,
                lr=0.7,
                nesterov=True,
                momentum=0.9)

        self.diloco_interval = diloco_interval


    def _average_models(self) -> None:
        for param in self.model.parameters():
            all_reduce(param.data, op=dist.ReduceOp.SUM)
            param.data /= self.num_nodes

    def _broadcast_model_params(self) -> None:
        for param in self.model.parameters():
            broadcast(param.data, src=0)

    def _set_master_grad(self) -> None:
        for name, param in self.master_model.named_parameters():
            param.grad = param.data - self.model.state_dict()[name].data.to('cpu')

    def _synchronize_master_model(self) -> None:
        for name, param in self.model.named_parameters():
            param.data = self.master_model.state_dict()[name].data.to(param.device)

    def step(self):
        if 'max_norm' in self.kwargs:
            nn_utils.clip_grad_norm_(self.model.parameters(), max_norm=self.kwargs['max_norm'])

        # We have just calculated the loss and done the backward pass. 
        # Therefore we do inner step first.
        self.optim.step()

        # Outer step if needed.
        if self.local_step % self.diloco_interval == 0 and self.local_step > 0:
            self._average_models()

            if self.rank == 0:
                self.outer_optimizer.zero_grad()
                self._set_master_grad()
                self.outer_optimizer.step()
                self._synchronize_master_model()

            self._broadcast_model_params()

        super().step()

    def _init_node(self, model, rank, num_nodes):
        super()._init_node(model, rank, num_nodes)


        if self.rank == 0:
            self.master_model = deepcopy(model).to("cpu")
            for param in self.master_model.parameters():
                param.requires_grad = True

            self.outer_optimizer = self.outer_optim_spec.build(self.master_model)

        self.optim = self.inner_optim_spec.build(model)
        self._setup_scheduler()