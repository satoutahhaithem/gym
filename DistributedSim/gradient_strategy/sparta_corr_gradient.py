import math
import torch
import torch.distributed as dist
from torch import nn
import os
import numpy as np
import wandb

from .gradient_strategy import GradientStrategy
from .sparta_gradient import RandomIndexSelector, PartitionedIndexSelector
from .communicate import *

class SPARTACorrelationGradient(GradientStrategy):
    def __init__(self, rank, model, config, logger=None):
        super().__init__(rank, model, config, logger)

        self.optim = self.gradient_config.optimizer_class(model.parameters(), 
                                                          **self.gradient_config.optimizer_kwargs)
        self._setup_scheduler()

        self.target_correlation = config.gradient_config.target_corr
        if self.rank == 0:
            self.p_sparta = self.gradient_config.p_sparta
            self.index_selector = RandomIndexSelector(self.p_sparta)

    def step(self):
        if self.gradient_config.max_norm:
            norm = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_config.max_norm)

        self.optim.step()

        if self.config.num_nodes > 1:
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if not param.requires_grad:
                        continue

                    if self.rank == 0:
                        indices = self.index_selector.get_indices(param)
                    else:
                        indices = torch.zeros_like(param, dtype=torch.bool)

                    # print('point 1')
                    # print(self.rank, indices)
                    broadcast(indices, src=0)
                    # print(self.rank, 'point 2')
                    sparse_data = param.data[indices]
                    all_reduce(sparse_data, op=dist.ReduceOp.SUM)
                    # print('point 3')
                    sparse_data /= dist.get_world_size()

                    param.masked_scatter_(indices, sparse_data)

        if self.local_step % self.gradient_config.rebalance_frequency == 0:# and self.local_step > 0:
            model_corr = self._correlation_calculation()

            if self.rank == 0:
                delta_corr = self.target_correlation - model_corr

                p_scale_factor = 1 + delta_corr * self.gradient_config.p_lr * self.gradient_config.rebalance_frequency

                # self.p_sparta = self.p_sparta * p_scale_factor
                self.index_selector.p = self.index_selector.p * p_scale_factor

                print(f'current correlation {model_corr} - updating p to {self.index_selector.p}')

        super().step()

    def _correlation_calculation(self):
        if self.config.num_nodes < 2:
            raise Exception('SPARTA cannot be used with < 2 nodes')
        # Create a temporary directory for this timestep's checkpoints
        tmp_dir = os.path.join(self.config.save_dir, f"tmp_corr_{self.local_step}")
        os.makedirs(tmp_dir, exist_ok=True)

        # Save model state dict for each rank
        checkpoint_path = os.path.join(tmp_dir, f"{self.rank}.pt")
        torch.save(self.model.state_dict(), checkpoint_path)

        # Wait for all processes to save their checkpoints
        torch.distributed.barrier()

        if self.rank == 0:
            # Load all models as vectors
            model_vectors = []
            for rank in range(self.config.num_nodes):
                model_path = os.path.join(tmp_dir, f"{rank}.pt")
                checkpoint = torch.load(model_path, map_location='cpu')
                vector_list = []
                for key in sorted(checkpoint.keys()):
                    value = checkpoint[key]
                    if isinstance(value, torch.Tensor):
                        vector_list.append(value.cpu().numpy().ravel())
                model_vectors.append(np.concatenate(vector_list))

            # Calculate correlations between all pairs
            correlations = []
            for i in range(self.config.num_nodes):
                for j in range(i+1, self.config.num_nodes):
                    corr = np.corrcoef(model_vectors[i], model_vectors[j])[0, 1]
                    correlations.append(corr)

            # # *** Additional variance logging: compute element-wise parameter variance across models.
            # stacked_vectors = np.stack(model_vectors, axis=0)  # shape: (num_nodes, total_params)
            # param_variance = np.var(stacked_vectors, axis=0)     # variance for each parameter element
            # avg_param_variance = np.mean(param_variance)

            corr_value = np.mean(correlations)            

            # Log average correlation and average parameter variance to wandb
            if self.config.wandb_project is not None:
                wandb.log(
                    {
                        "avg_model_correlation": corr_value,
                        # "avg_parameter_variance": avg_param_variance,
                    },
                    step=self.local_step,
                )

            # Clean up temporary directory
            import shutil
            shutil.rmtree(tmp_dir)

        # Wait for rank 0 to finish cleanup
        torch.distributed.barrier()

        if self.rank == 0:
            return corr_value
        else:
            return None