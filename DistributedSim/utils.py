import torch
import numpy as np
from typing import Any, Dict, Union, List


def extract_wandb_config(obj, max_depth=3, current_depth=0):
  """
  Extract serializable configuration from an object for wandb logging.
  
  This function safely extracts attributes from objects while avoiding
  unpickleable items like tensors, functions, modules, etc.
  
  Args:
    obj: The object to extract config from
    max_depth: Maximum recursion depth to avoid infinite loops
    current_depth: Current recursion depth (internal use)
    
  Returns:
    dict: A dictionary containing only serializable attributes
  """
  if current_depth >= max_depth:
    return str(type(obj).__name__)
    
  if obj is None:
    return None
    
  # Handle primitive types
  if isinstance(obj, (int, float, str, bool)):
    return obj
    
  # Handle sequences (but avoid strings which are also sequences)
  if isinstance(obj, (list, tuple)) and not isinstance(obj, str):
    try:
      return [extract_wandb_config(item, max_depth, current_depth + 1) for item in obj[:10]]  # Limit to first 10 items
    except:
      return f"<{type(obj).__name__} with {len(obj)} items>"
      
  # Handle dictionaries
  if isinstance(obj, dict):
    try:
      result = {}
      for key, value in obj.items():
        if isinstance(key, str) and len(result) < 50:  # Limit number of keys
          result[key] = extract_wandb_config(value, max_depth, current_depth + 1)
      return result
    except:
      return f"<dict with {len(obj)} items>"
      
  # Skip unpickleable types
  if isinstance(obj, (torch.Tensor, torch.nn.Module, torch.optim.Optimizer, 
                     torch.nn.Parameter, torch.dtype, torch.device)):
    if isinstance(obj, torch.Tensor):
      return f"<Tensor {list(obj.shape)}>"
    elif isinstance(obj, torch.nn.Module):
      return f"<Module {type(obj).__name__}>"
    elif isinstance(obj, torch.optim.Optimizer):
      return f"<Optimizer {type(obj).__name__}>"
    else:
      return f"<{type(obj).__name__}>"
      
  # Skip functions, methods, and other callables
  if callable(obj):
    return f"<function {getattr(obj, '__name__', 'unknown')}>"
    
  # Handle objects with __dict__ (like config objects)
  if hasattr(obj, '__dict__'):
    try:
      result = {}
      for key, value in obj.__dict__.items():
        if not key.startswith('_') and len(result) < 50:  # Skip private attributes
          try:
            result[key] = extract_wandb_config(value, max_depth, current_depth + 1)
          except:
            result[key] = f"<error extracting {key}>"
      return result
    except:
      return f"<{type(obj).__name__} object>"
      
  # For other objects, try to get basic info
  try:
    return f"<{type(obj).__name__}>"
  except:
    return "<unknown object>"


def create_wandb_config(model: torch.nn.Module, 
                       strategy=None, 
                       extra_config: Dict[str, Any] = None) -> Dict[str, Any]:
  """
  Create a comprehensive wandb configuration from model, strategy, and config objects.
  
  Args:
    model: The PyTorch model
    strategy: The training strategy object (optional)
    config: The main configuration object (optional)
    extra_config: Additional configuration to include (optional)
    
  Returns:
    dict: A complete wandb configuration dictionary
  """
  ## TODO: Fix this up to log everything we need.
  wandb_config = {}
  
  # Model information
  if model:
    wandb_config.update({
      "model_name": model.__class__.__name__,
      "model_config": extract_wandb_config(model),
    })
    
    # Try to get parameter count
    try:
      if hasattr(model, 'get_num_params'):
        wandb_config["model_parameters"] = model.get_num_params() / 1e6
      else:
        # Fallback to counting parameters
        wandb_config["model_parameters"] = sum(p.numel() for p in model.parameters()) / 1e6
    except:
      wandb_config["model_parameters"] = "unknown"
  
  # Strategy information
  if strategy:
    wandb_config.update({
      "strategy_name": strategy.__class__.__name__,
      "strategy_config": extract_wandb_config(strategy),
    })

  # # Main configuration
  # if config:
  #   config_dict = extract_wandb_config(config)
  #   # Remove potentially problematic keys
  #   keys_to_remove = ['model_class', 'train_dataset', 'val_dataset', 'model']
  #   for key in keys_to_remove:
  #     if key in config_dict:
  #       del config_dict[key]
  #   wandb_config.update(config_dict)

  # Extra configuration
  if extra_config:
    for key, value in extra_config.items():
      wandb_config[key] = extract_wandb_config(value)
  
  return wandb_config


def log_model_summary(model: torch.nn.Module) -> Dict[str, Any]:
  """
  Create a summary of model architecture suitable for logging.
  
  Args:
    model: The PyTorch model
    
  Returns:
    dict: Model summary information
  """
  summary = {
    "model_class": model.__class__.__name__,
    "model_module": model.__class__.__module__,
  }
  
  try:
    # Parameter count
    if hasattr(model, 'get_num_params'):
      summary["total_params"] = model.get_num_params()
    else:
      summary["total_params"] = sum(p.numel() for p in model.parameters())
    
    summary["total_params_M"] = summary["total_params"] / 1e6
    
    # Trainable parameters
    summary["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    summary["trainable_params_M"] = summary["trainable_params"] / 1e6
    
    # Model config if available
    if hasattr(model, 'config'):
      summary["config"] = extract_wandb_config(model.config)
    
    # Layer information
    layer_types = {}
    for name, module in model.named_modules():
      module_type = type(module).__name__
      if module_type != model.__class__.__name__:  # Skip the root module
        layer_types[module_type] = layer_types.get(module_type, 0) + 1
    summary["layer_types"] = layer_types
    
  except Exception as e:
    summary["error"] = f"Error extracting model summary: {str(e)}"
  
  return summary


def safe_log_dict(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
  """
  Convert a dictionary to a wandb-safe format.
  
  Args:
    data: Dictionary to convert
    prefix: Prefix to add to keys
    
  Returns:
    dict: Wandb-safe dictionary
  """
  safe_dict = {}
  
  for key, value in data.items():
    safe_key = f"{prefix}_{key}" if prefix else key
    safe_dict[safe_key] = extract_wandb_config(value)
  
  return safe_dict 