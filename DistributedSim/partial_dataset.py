import torch

## TODO: Should we be using DistributedSampler instead?
## This definitely isn't the ideal way to do this because it's still copying the whole dataset per node.
class PartialDataset(torch.utils.data.Dataset):
  def __init__(self, dataset: torch.utils.data.Dataset, rank: int, num_nodes: int):
    self.dataset = dataset
    self.rank = rank
    self.num_nodes = num_nodes

  def __len__(self):
    return len(self.dataset) // self.num_nodes

  def __getitem__(self, idx):
    return self.dataset[idx * self.num_nodes + self.rank]


# class PartialDataset(IterableDataset):
#   def __init__(self, dataset, rank: int, num_nodes: int):
#     self.base_dataset = dataset
#     self.rank = rank
#     self.num_nodes = num_nodes

#   def __len__(self):
#     return len(self.dataset) // self.num_nodes

#   def __iter__(self):
#     for idx, sample in enumerate(self.base_dataset):
#       if idx % self.num_nodes == self.rank:
#         yield sample 