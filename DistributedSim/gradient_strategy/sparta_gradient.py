import math
import torch
import torch.distributed as dist
from torch import nn

from .gradient_strategy import GradientStrategy
from .communicate import *

class SPARTAGradient(GradientStrategy):
    def __init__(self, rank, model, config, logger=None):
        super().__init__(rank, model, config, logger)

        self.optim = self.gradient_config.optimizer_class(model.parameters(), 
                                                          **self.gradient_config.optimizer_kwargs)
        self._setup_scheduler()

        # self.index_selector = PartitionedIndexSelector(self.gradient_config.p_sparta)
        # self.index_selector = RandomIndexSelector(self.gradient_config.p_sparta)
        # self.index_selector = ShuffledSequentialIndexSelector(self.gradient_config.p_sparta)

        print(self.gradient_config.__dict__)        

        if hasattr(self.gradient_config, 'param_weights'):
            print('weighted!')
            self.index_selector = RandomWeightedIndexSelector(self.gradient_config.p_sparta, self.gradient_config.param_weights, model)
        else:
            print('unweighted :(')
            self.index_selector = ShuffledSequentialIndexSelector(self.gradient_config.p_sparta)

        self.iteration = 0
        self.buffer = {} # Initialize as a dictionary for per-parameter buffers
        self.fault_rate = getattr(self.gradient_config, 'fault_rate', 0.0)

        if self.gradient_config.schedule_p:
            self._calculate_p_schedule_params()

    def _calculate_p_schedule_params(self):
        p_avg = self.gradient_config.p_sparta
        p_min_target_factor = self.gradient_config.p_min_factor
        
        # Ensure warmup_steps and max_local_steps are available and valid
        warmup_steps = float(getattr(self.gradient_config, 'warmup_steps', 0))
        max_steps = float(getattr(self.gradient_config, 'max_local_steps', 1))

        if max_steps <= 0:
            self.p_sched_initial_value = p_avg 
            self.p_sched_final_value = p_avg
            if self.rank == 0 and hasattr(self.logger, 'log') and callable(self.logger.log):
                self.logger.log({"sparta_p_schedule_warning": "max_steps <= 0, using constant p_avg"})
            return

        if warmup_steps < 0: warmup_steps = 0
        if warmup_steps > max_steps: warmup_steps = max_steps

        target_final_p = p_avg * p_min_target_factor

        # Let T_w = warmup_steps, T_m = max_steps, T_a = max_steps - warmup_steps (anneal duration)
        # p_avg * T_m = initial_p * T_w + (initial_p + target_final_p)/2 * T_a
        # initial_p * (T_w + T_a/2) = p_avg * T_m - target_final_p * T_a / 2
        # initial_p * ( (2*T_w + T_a)/2 ) = (2 * p_avg * T_m - target_final_p * T_a) / 2
        # initial_p = (2 * p_avg * T_m - target_final_p * T_a) / (2 * T_w + T_a)
        # Denominator: 2 * T_w + T_a = T_w + T_w + T_m - T_w = T_w + T_m
        denominator = warmup_steps + max_steps
        anneal_duration = max_steps - warmup_steps

        if abs(denominator) < 1e-9: # Should be caught by max_steps <= 0, but as a safeguard
            initial_p_candidate = p_avg
        else:
            initial_p_candidate = (2 * p_avg * max_steps - target_final_p * anneal_duration) / denominator
        
        calculated_final_p = target_final_p

        if initial_p_candidate > 1.0:
            self.p_sched_initial_value = 1.0
            if anneal_duration <= 1e-9: 
                calculated_final_p = self.p_sched_initial_value 
            else:
                # p_avg * T_m = 1.0 * T_w + (1.0 + final_p_recalc)/2 * T_a
                # (2 * (p_avg * T_m - T_w) / T_a) - 1.0 = final_p_recalc
                calculated_final_p = (2 * (p_avg * max_steps - warmup_steps) / anneal_duration) - 1.0
        elif initial_p_candidate < 0.0:
            self.p_sched_initial_value = 0.0
            if anneal_duration <= 1e-9:
                calculated_final_p = self.p_sched_initial_value
            else:
                # p_avg * T_m = 0.0 * T_w + (0.0 + final_p_recalc)/2 * T_a
                # 2 * p_avg * T_m / T_a = final_p_recalc
                calculated_final_p = (2 * p_avg * max_steps) / anneal_duration
        else:
            self.p_sched_initial_value = initial_p_candidate
            # calculated_final_p remains target_final_p

        self.p_sched_initial_value = min(1.0, max(0.0, self.p_sched_initial_value))
        calculated_final_p = min(1.0, max(0.0, calculated_final_p))
        
        self.p_sched_final_value = min(calculated_final_p, self.p_sched_initial_value)

        if self.rank == 0 and hasattr(self.logger, 'log') and callable(self.logger.log):
            print(f"sparta_p_sched_initial: {self.p_sched_initial_value}, sparta_p_sched_final: {self.p_sched_final_value}, sparta_p_avg_target: {p_avg}")

    def _get_current_p(self):
        # Ensure warmup_steps and max_local_steps are available and valid
        warmup_steps = float(getattr(self.gradient_config, 'warmup_steps', 0))
        max_steps = float(getattr(self.gradient_config, 'max_local_steps', 1))
        current_step = float(self.iteration)

        if max_steps <= 0: return self.gradient_config.p_sparta # Should use calculated if available
        if warmup_steps < 0: warmup_steps = 0
        if warmup_steps > max_steps: warmup_steps = max_steps

        initial_p = self.p_sched_initial_value
        final_p = self.p_sched_final_value

        if current_step < warmup_steps:
            return initial_p
        elif current_step < max_steps:
            anneal_duration = max_steps - warmup_steps
            if anneal_duration <= 1e-9: # Should be covered by current_step < warmup_steps or current_step >= max_steps
                return initial_p # Or final_p, effectively constant during this phase
            
            progress = (current_step - warmup_steps) / anneal_duration
            cosine_term = 0.5 * (1.0 + math.cos(math.pi * progress))
            current_p_val = final_p + (initial_p - final_p) * cosine_term
            return current_p_val
        else: 
            return final_p

    def step(self):
        if self.gradient_config.max_norm:
            norm = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_config.max_norm)
            # print(f'Rank {self.rank}: Clipped grad norm to {norm}')

        self.optim.step()

        if self.gradient_config.schedule_p:
            current_p_for_comm = self._get_current_p()
            self.index_selector.p = current_p_for_comm # Update p in index selector
            if self.rank == 0 and hasattr(self.logger, 'log') and callable(self.logger.log):
                self.logger.log({'sparta_p': current_p_for_comm})
        else:
            current_p_for_comm = self.gradient_config.p_sparta

        simulate_fault_this_round = False # Initialize default

        if self.config.num_nodes > 1:
            # Synchronized decision for fault simulation
            # Ensure the decision tensor is on the same device as model parameters
            device = next(self.model.parameters()).device 
            decision_tensor = torch.zeros(1, dtype=torch.float32, device=device) # 0.0 for no fault, 1.0 for fault

            if self.rank == 0:
                if self.fault_rate > 0.0 and torch.rand(1).item() < self.fault_rate:
                    decision_tensor[0] = 1.0 # Mark for fault simulation
            
            # Broadcast the decision from rank 0 to all other ranks.
            # This is a collective operation and MUST be called by all ranks in the group.
            broadcast(decision_tensor, src=0) 
            
            # All ranks now have the same decision
            if decision_tensor.item() == 1.0:
                simulate_fault_this_round = True
            
            # The following conditional block, modified by the user, will now execute
            # synchronously based on the globally decided 'simulate_fault_this_round'.
            if not simulate_fault_this_round:
                # Proceed with normal communication and updates if no fault is simulated
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if not param.requires_grad or param.grad is None:
                            continue

                        indices_mask = self.index_selector.get_indices(param, self.iteration)

                        ## TODO: Apparently this doesn't work well with non-contiguous data
                        broadcast(indices_mask, src=0) # Broadcasting a mask might be needed
                        sparse_data = param.data[indices_mask] # Get data using the mask
                        all_reduce(sparse_data, op=dist.ReduceOp.SUM) # This likely won't work as expected with masked, non-contiguous data
                        sparse_data /= dist.get_world_size()

                        # Initialize buffer for this parameter if it doesn't exist
                        if name not in self.buffer:
                            self.buffer[name] = []
                        
                        # Add current sparse update to this parameter's buffer
                        self.buffer[name].append((indices_mask, sparse_data))

                        # If this parameter's buffer has exceeded the delay, apply the oldest update
                        if len(self.buffer[name]) > self.gradient_config.async_sparta_delay:
                            indices_popped, sparse_data_popped = self.buffer[name].pop(0)
                            # Apply the popped update to the current parameter (param corresponds to name)
                            param.masked_scatter_(indices_popped, sparse_data_popped)

        # Increment iteration AFTER potentially using it for gradient updates/communication
        self.iteration += 1
        super().step() # This likely calls scheduler.step()

class IndexSelector:
    def __init__(self, p):
        self.state = {}
        self.p = p

    # Add iteration argument to the base class signature
    def get_indices(self, param, iteration):
        # Default implementation returns all indices (mask of Trues)
        return torch.ones_like(param, dtype=torch.bool)

class RandomIndexSelector(IndexSelector):
    def get_indices(self, param, iteration):
        return torch.bernoulli(torch.full(param.shape, self.p, device=param.device)).bool()
class RandomWeightedIndexSelector(IndexSelector):
  def __init__(self, p_base, param_weights, model):
    """
    Initializes the RandomWeightedIndexSelector.

    Args:
      p_base (float): The base probability for an element to be selected,
                      if all weights were 1.0. This is the target average
                      proportion of elements to be selected across the entire model.
      param_weights (dict): A dictionary where keys are substrings of parameter names
                            (e.g., 'encoder', 'layer.0.weight') and values are
                            multiplicative weights for those parameters' selection
                            probabilities. Parameters not matching any key get a
                            default weight of 1.0. More specific (longer) keys
                            are prioritized if names match multiple keys.
      model (torch.nn.Module): The model whose parameters will be sampled.
    """
    super().__init__(p_base)  # Initializes self.p = p_base and self.state = {}
    self.p_base = p_base
    self._param_probs = {}  # Stores param_id -> calculated_probability

    parameter_details = []
    total_numel_model = 0

    # Collect details for each parameter
    for name, param in model.named_parameters():
      if param.numel() == 0:  # Skip zero-element tensors
        continue
      
      total_numel_model += param.numel()
      
      current_param_weight = 1.0  # Default weight
      
      effective_param_weights = param_weights if param_weights is not None else {}

      # Determine weight: prefer longer (more specific) key matches
      # Sort keys by length (descending)
      sorted_weight_keys = sorted(effective_param_weights.keys(), key=len, reverse=True)

      for key_pattern in sorted_weight_keys:
        if key_pattern in name:
          current_param_weight = effective_param_weights[key_pattern]
          break  # Found the most specific match
      
      parameter_details.append({
        'name': name, 
        'param': param,
        'weight': current_param_weight,
        'numel': param.numel()
      })

    sum_of_weighted_numels = 0.0
    for detail in parameter_details:
      sum_of_weighted_numels += detail['weight'] * detail['numel']

    normalization_factor_C = 1.0
    if total_numel_model > 0:
      if abs(sum_of_weighted_numels) > 1e-9: # Avoid division by zero or instability
        normalization_factor_C = total_numel_model / sum_of_weighted_numels
      # If sum_of_weighted_numels is zero (e.g. all weights are zero), C remains 1.0.
      # p_param_specific will become p_base * weight * 1.0. If weight is 0, prob is 0.

    for detail in parameter_details:
      param = detail['param']
      weight = detail['weight']
      
      p_param_specific = self.p_base * weight * normalization_factor_C
      p_param_specific = max(0.0, min(1.0, p_param_specific)) # Clamp to [0, 1]
      
      self._param_probs[id(param)] = p_param_specific

  def get_indices(self, param, iteration):
    """
    Gets a boolean mask for selecting indices from a given parameter.

    Args:
      param (torch.Tensor): The parameter tensor for which to generate a mask.
      iteration (int): The current training iteration (unused by this selector).

    Returns:
      torch.Tensor: A boolean tensor of the same shape as `param`, where True
                    indicates an element should be selected.
    """
    if param.numel() == 0:
      return torch.zeros_like(param, dtype=torch.bool)

    param_id = id(param)
    
    if param_id not in self._param_probs:
      # Fallback for params not found (e.g. if model changed post-init)
      return torch.zeros_like(param, dtype=torch.bool)
    
    p_for_this_param = self._param_probs[param_id]
    
    prob_tensor = torch.full_like(param, 
                                  fill_value=float(p_for_this_param), 
                                  dtype=torch.float32, 
                                  device=param.device)
    return torch.bernoulli(prob_tensor).bool()

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

        # self.p is updated by SPARTAGradient before calling get_indices
        # Handle p near zero to avoid division by zero or extreme num_partitions
        if self.p <= 1e-9: 
            return torch.zeros_like(param, dtype=torch.bool) # Select nothing if p is effectively zero

        # num_partitions is calculated dynamically based on the current self.p
        # Ensure self.p is positive before division
        current_num_partitions = max(1, math.ceil(1.0 / self.p)) 

        if param not in self.state:
            # shuffled_indices are generated once per parameter and stored.
            shuffled_indices_val = torch.randperm(num_total, device=param.device)
            self.state[param] = {
                "shuffled_indices": shuffled_indices_val,
            }
        
        param_state = self.state[param]
        shuffled_indices = param_state["shuffled_indices"]

        # Determine the current chunk based on the iteration number
        # current_chunk must be less than current_num_partitions
        current_chunk = iteration % current_num_partitions

        # Calculate chunk size and remainder for potentially uneven distribution
        chunk_size = num_total // current_num_partitions
        remainder = num_total % current_num_partitions

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
