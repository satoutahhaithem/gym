import torch
import numpy as np

class NonContiguousGPTTrainDataset(torch.utils.data.Dataset):
    """Dataset for pre-segmented training data (2D tensor).
    Each row is an independent sequence with no continuity between examples.
    Suitable for datasets already divided into fixed-length chunks.
    """
    def __init__(self, data, storage_device, compute_device):
        assert data.ndim == 2
        self.examples, self.block_size = data.shape

        self.storage_device = storage_device
        self.compute_device = compute_device

        self.data = torch.from_numpy(data).to(device=storage_device).long()

    def __len__(self):
        return self.examples

    def __getitem__(self, idx):
        x = self.data[idx].to(self.compute_device)
        return x[:-1], x[1:]

class ContiguousGPTTrainDataset(torch.utils.data.Dataset):
    """Dataset for continuous token streams (1D tensor).
    Creates examples by sliding a window over the data.
    Preserves context and long-range dependencies in text.
    """
    def __init__(self, data, block_size, storage_device, compute_device):
        assert data.ndim == 1

        self.storage_device = storage_device
        self.compute_device = compute_device

        self.data = torch.from_numpy(data).to(device=storage_device).long()
        self.block_size = block_size

    def __len__(self):
        return self.data.shape[0] - self.block_size - 1

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.block_size + 1].to(self.compute_device)
        return x[:-1], x[1:]