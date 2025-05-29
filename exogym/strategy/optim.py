import torch

from typing import Type

from dataclasses import dataclass 
from typing import Dict, Any

@dataclass
class OptimSpec:
    cls:  Type[torch.optim.Optimizer]   = torch.optim.AdamW
    kwargs: Dict[str, Any]              = None          # e.g. {'lr': 3e-4}

    def __init__(self, cls: Type[torch.optim.Optimizer], **kwargs: Dict[str, Any]):
        self.cls = cls
        self.kwargs = kwargs

    def build(self, model):
        return self.cls(model.parameters(), **(self.kwargs or {}))