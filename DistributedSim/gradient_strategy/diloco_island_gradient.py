import torch.distributed as dist
from copy import deepcopy
import random

import torch
from torch.nn import utils as nn_utils

from .gradient_strategy import GradientStrategy
from .communicate import *

class DiLoCoIslandGradient(GradientStrategy):
    def __init__(self, rank, model, config, logger=None):
        super().__init__(rank, model, config, logger)

        self.local_step = 0

        device = next(model.parameters()).device
        self.master_model = deepcopy(model).to(device)
        for param in self.master_model.parameters():
            param.requires_grad = True

        self.outer_optimizer = self.gradient_config.outer_optimizer_cls(self.master_model.parameters(), 
                                                                        **self.gradient_config.outer_optimizer_kwargs)

        self.optim = self.gradient_config.optimizer_class(model.parameters(), 
                                                          **self.gradient_config.optimizer_kwargs)
        self._setup_scheduler()

    def _select_partners(self):
        world_size = dist.get_world_size()
        
        # Only rank 0 creates the partner assignments.
        partners = None
        if self.rank == 0:
            # Create a list of all rank numbers and shuffle it.
            ranks = list(range(world_size))
            ## TODO: Switch to pytorch shuffle.
            random.shuffle(ranks)
            
            # Initialize partner list where index is the rank and value is its partner.
            partners = [-1] * world_size
            
            # Pair off ranks in order.
            while len(ranks) >= 2:
                a = ranks.pop(0)
                b = ranks.pop(0)
                partners[a] = b
                partners[b] = a
            # If there's an odd number, one rank remains unpaired (indicated by -1).
        
        # Broadcast the partner list from rank 0 to all processes.
        partners_list = [partners] if self.rank == 0 else [None]
        dist.broadcast_object_list(partners_list, src=0)
        partners = partners_list[0]

        print(f'Rank {self.rank} has partner {partners[self.rank]}')
        
        return partners[self.rank]

    def _average_models(self, partner_rank) -> None:
        ## Average *local* model parameters
        for param in self.model.parameters():
            tensor_list = [torch.zeros_like(param.data) for _ in range(self.config.num_nodes)]
            all_gather(tensor_list, param.data)
            param.data = (tensor_list[self.rank] + tensor_list[partner_rank]) / 2

    # def _set_master_grad(self) -> None:
    #     for name, param in self.model.named_parameters():
    #         param.grad = self.master_model.state_dict()[name].data.to(param.device) - param.data

    def _set_master_grad(self) -> None:
        for name, param in self.master_model.named_parameters():
            param.grad = param.data - self.model.state_dict()[name].data

    def _synchronize_master_model(self) -> None:
        # Save updated master model parameters to model.
        for name, param in self.model.named_parameters():
            param.data = self.master_model.state_dict()[name].data

    def step(self):
        if self.gradient_config.max_norm:
            nn_utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_config.max_norm)

        # We have just calculated the loss and done the backward pass. 
        # Therefore we do inner step first.
        self.optim.step()

        # Outer step if needed.
        if self.local_step % self.gradient_config.diloco_interval == 0 and self.local_step > 0:
            partner_rank = self._select_partners()
            self._average_models(partner_rank)

            self.outer_optimizer.zero_grad()
            self._set_master_grad()
            self.outer_optimizer.step()

            self._synchronize_master_model()

        super().step()

        self.local_step += 1