import torch.distributed as dist
from copy import deepcopy
import random

import torch
from torch.nn import utils as nn_utils

from typing import Optional

from .strategy import Strategy
from .optim import OptimSpec
from .communicate import *

class FedAvgStrategy(Strategy):
    def __init__(self, 
                 optim_spec: OptimSpec,
                 island_size: Optional[int] = None,
                 H: int = 1,
                 max_norm: float = None,
                 **kwargs):
        super().__init__(**kwargs)

        self.optim_spec = optim_spec
        self.island_size = island_size
        self.H = H
        self.max_norm = max_norm

    def _select_partners(self):
        """
        Selects partners for goruped Federated Averaging. By default not used.
        """
        world_size = dist.get_world_size()
        
        # Only rank 0 creates the island assignments
        islands = None
        if self.rank == 0:
            # Create a list of all rank numbers and shuffle it.
            ranks = list(range(world_size))
            ## TODO: Switch to pytorch shuffle.
            random.shuffle(ranks)
        else:
            ## TODO: Switch to pytorch broadcast.
            ranks = [None] * world_size

        dist.broadcast_object_list(ranks, src=0)

        islands = []
        for i in range(0, len(ranks), self.island_size):
            islands.append(set(ranks[i:i+self.island_size]))
        
        # Ugh seems so unoptimal but it's fine for now.
        my_island = None
        for island in islands:
            if self.rank in island:
                my_island = island
                break
        
        # print(f'Rank {self.rank} has partners {my_island}')
        
        return my_island

    def _average_models(self, island_members) -> None:
        ## Average model parameters across all members in the island
        for param in self.model.parameters():
            ## At the moment we are doing a full all_gather - this will be optimized in a full-scale training implementation.
            tensor_list = [torch.zeros_like(param.data) for _ in range(self.num_nodes)]
            all_gather(tensor_list, param.data)
            
            # Compute average only from ranks in the same island
            island_tensors = [tensor_list[rank] for rank in island_members]
            island_average = sum(island_tensors) / len(island_tensors)
            
            param.data = island_average

    def step(self):
        if self.max_norm:
            nn_utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm)

        # We have just calculated the loss and done the backward pass. 
        # Therefore we do inner step first.
        self.optim.step()

        # Outer step if needed.
        if self.local_step % self.H == 0 and self.local_step > 0:
            if self.island_size < self.num_nodes:
                island_members = self._select_partners()
            else:
                island_members = list(range(self.num_nodes))

            self._average_models(island_members)

        super().step()

    def _init_node(self, model, rank, num_nodes):
        super()._init_node(model, rank, num_nodes)

        if self.island_size is None:
            self.island_size = num_nodes

        self.optim = self.optim_spec.build(model)
        self._setup_scheduler()