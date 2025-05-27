import math
import torch
import torch.distributed as dist
from torch import nn
import copy

from typing import Dict, Any

from .strategy import Strategy
from .optim import OptimSpec
from .communicate import *

class SPARTAStrategy(Strategy):
    def __init__(self, 
                 optim_spec: OptimSpec,
                 lr_scheduler: str = None,
                 lr_scheduler_kwargs: Dict[str, Any] = None,
                 p_sparta=0.005):

        super().__init__(lr_scheduler=lr_scheduler,
                         lr_scheduler_kwargs=lr_scheduler_kwargs)

        self.optim_spec = optim_spec

        self.index_selector = RandomIndexSelector(p_sparta)
        # self.index_selector = ShuffledSequentialIndexSelector(p_sparta)
        self.iteration = 0

    def step(self):
        # if self.strategy_config.max_norm:
        #     norm = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.strategy_config.max_norm)

        self.optim.step()

        if self.num_nodes > 1:
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if not param.requires_grad or param.grad is None:
                        continue

                    indices_mask = self.index_selector.get_indices(param, self.iteration)

                    ## TODO: Apparently this doesn't work well with non-contiguous data
                    broadcast(indices_mask, src=0) # Broadcasting a mask might be needed
                    sparse_data = param.data[indices_mask] # Get data using the mask
                    all_reduce(sparse_data, op=dist.ReduceOp.SUM) # This likely won't work as expected with masked, non-contiguous data
                    sparse_data /= self.num_nodes

                    param.masked_scatter_(indices_mask, sparse_data)

        self.iteration += 1
        super().step()

    def _init_node(self, model, rank, num_nodes):
        super()._init_node(model, rank, num_nodes)

        self.optim = self.optim_spec.build(model)

        self._setup_scheduler()

class IndexSelector:
    def __init__(self, p):
        self.state = {}
        self.p = p

    # Add iteration argument to the base class signature
    def get_indices(self, param, iteration):
        # Default implementation returns all indices (mask of Trues)
        return torch.ones_like(param, dtype=torch.bool)


class RandomIndexSelector(IndexSelector):
    # Update signature to match base class
    def get_indices(self, param, iteration):
        return torch.bernoulli(torch.full(param.shape, self.p, device=param.device)).bool()

class ShuffledSequentialIndexSelector(IndexSelector):
    def __init__(self, p):
        # No model-dependent init here
        super().__init__(p)
        # Remove self.shuffled_state and self.index

    # Update signature to match base class
    def get_indices(self, param, iteration):
        num_total = param.numel()
        if num_total == 0:
            return torch.zeros_like(param, dtype=torch.bool)

        # Initialize state for this parameter if not seen before
        if param not in self.state:
            num_partitions = max(1, math.ceil(1.0 / self.p)) # Ensure at least 1 partition
            shuffled_indices = torch.randperm(num_total, device=param.device)
            self.state[param] = {
                "num_partitions": num_partitions,
                "shuffled_indices": shuffled_indices,
            }

        param_state = self.state[param]
        num_partitions = param_state["num_partitions"]
        shuffled_indices = param_state["shuffled_indices"]

        # Determine the current chunk based on the iteration number
        current_chunk = iteration % num_partitions

        # Calculate chunk size and remainder for potentially uneven distribution
        chunk_size = num_total // num_partitions
        remainder = num_total % num_partitions

        # Calculate start and end indices for the current chunk
        start_index = current_chunk * chunk_size + min(current_chunk, remainder)
        # The end index calculation ensures the chunk size is correct, adding 1 for chunks getting the remainder
        end_index = start_index + chunk_size + (1 if current_chunk < remainder else 0)

        # Get the flat indices for the current chunk
        selected_flat_indices = shuffled_indices[start_index:end_index]

        # Create and return the boolean mask
        mask = torch.zeros(num_total, dtype=torch.bool, device=param.device)
        if selected_flat_indices.numel() > 0: # Handle empty selection if num_total is very small
            mask[selected_flat_indices] = True
        return mask.view(param.shape)


class PartitionedIndexSelector(IndexSelector):
    def __init__(self, p):
        super().__init__(p)
        # Note: This class implicitly uses a step counter per parameter via self.state[param]["curr_partition"]
        # It doesn't need the global iteration number passed in.
        # To be consistent, we should update its signature, but the iteration argument would be unused.

    def _set_partition(self, param):
        param_state = self.state[param]
        param_state["curr_partition"] = 0
        # Ensure at least 1 partition
        num_partitions = max(1, min(math.ceil(1.0 / self.p), param.numel()))
        param_state["num_partitions"] = num_partitions
        if param.numel() > 0:
            param_state["partitions"] = (
                torch.rand(param.numel(), device=param.device).argsort() % num_partitions
            )
        else:
            # Handle zero-element tensors
            param_state["partitions"] = torch.empty(0, dtype=torch.long, device=param.device)


    # Update signature, though iteration is unused here
    def get_indices(self, param, iteration):
        # Handle zero-element tensors gracefully
        if param.numel() == 0:
            return torch.zeros_like(param, dtype=torch.bool)

        if param not in self.state:
            self.state[param] = {}
            self._set_partition(param)
        # Check if cycle needs reset BEFORE accessing partitions
        elif self.state[param]["curr_partition"] >= self.state[param]["num_partitions"]:
            self._set_partition(param)

        param_state = self.state[param]
        
        # Need to handle case where num_partitions might be 0 if numel was 0 during _set_partition
        # Although we added checks for numel=0, ensure partition access is safe
        if param_state["num_partitions"] == 0:
            return torch.zeros_like(param, dtype=torch.bool) # Should not happen if numel > 0


        # Indices calculation requires reshaping the flat partitions result
        partition_indices = (param_state["partitions"] == param_state["curr_partition"])
        indices_mask = partition_indices.view(param.shape).bool() # Reshape flat bool tensor to param shape

        param_state["curr_partition"] += 1

        return indices_mask