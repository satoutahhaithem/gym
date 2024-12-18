from dataclasses import dataclass, field
from typing import Optional, Callable, Type
import torch

class SimConfig:
    num_nodes: int

    model_class: Type[torch.nn.Module]
    model_kwargs: dict = {}

    dataset: torch.utils.data.Dataset
    batch_size: int = 32

    optimizer_class: Type[torch.optim.Optimizer] = torch.optim.SGD
    optimizer_kwargs: dict = {}

    criterion_class: Type[torch.nn.Module]
    criterion_kwargs: dict = {}
