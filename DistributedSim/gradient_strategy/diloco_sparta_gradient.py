import torch.distributed as dist
from copy import deepcopy

from torch.nn import utils as nn_utils
import torch

from .sparta_gradient import RandomIndexSelector
from .gradient_strategy import GradientStrategy
from .communicate import *

class DiLoCoSPARTAGradient(GradientStrategy):
    def __init__(self, rank, model, config, logger=None):
        super().__init__(rank, model, config, logger)

        if self.rank == 0:
            self.master_model = deepcopy(model).to("cpu")
            for param in self.master_model.parameters():
                param.requires_grad = True

            self.outer_optimizer = self.gradient_config.outer_optimizer_cls(self.master_model.parameters(), 
                                                                            **self.gradient_config.outer_optimizer_kwargs)

        self.optim = self.gradient_config.optimizer_class(model.parameters(), 
                                                          **self.gradient_config.optimizer_kwargs)
        self._setup_scheduler()

        self.index_selector = RandomIndexSelector(self.gradient_config.p_sparta)

    def _average_models(self) -> None:
        for param in self.model.parameters():
            all_reduce(param.data, op=dist.ReduceOp.SUM)
            param.data /= self.config.num_nodes

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
        if self.gradient_config.max_norm:
            nn_utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_config.max_norm)

        # We have just calculated the loss and done the backward pass. 
        # Therefore we do inner step first.
        self.optim.step()

        if self.config.num_nodes > 1 and self.local_step % self.gradient_config.sparta_interval == 0:
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if not param.requires_grad:
                        continue

                    indices = self.index_selector.get_indices(param, self.local_step)
                    broadcast(indices, src=0)
                    sparse_data = param.data[indices]
                    all_reduce(sparse_data, op=dist.ReduceOp.SUM)
                    sparse_data /= dist.get_world_size()

                    param.masked_scatter_(indices, sparse_data)

        # Outer step if needed.
        if self.local_step % self.gradient_config.diloco_interval == 0 and self.local_step > 0:
            self._average_models()

            if self.rank == 0:
                self.outer_optimizer.zero_grad()
                self._set_master_grad()
                self.outer_optimizer.step()
                self._synchronize_master_model()

            self._broadcast_model_params()

        super().step()