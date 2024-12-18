from dataclasses import dataclass, field
from typing import Optional, Callable, Type
import torch

from gradient_strategy import *

class SimConfig:
    num_nodes: int

    model_class: Type[torch.nn.Module]
    model_kwargs: dict = {}

    train_dataset: torch.utils.data.Dataset
    val_dataset: torch.utils.data.Dataset
    batch_size: int = 64

    optimizer_class: Type[torch.optim.Optimizer] = torch.optim.SGD
    optimizer_kwargs: dict = {}

    criterion_class: Type[torch.nn.Module]
    criterion_kwargs: dict = {}

    gradient_class: Type[GradientStrategy]
