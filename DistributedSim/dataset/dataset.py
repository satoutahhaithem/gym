from typing import Callable

class DatasetConfig:
    def __init__(self, 
                 dataset_name: str,
                 batch_size: int,
                 dataset_load_fn: Callable,
                 **kwargs):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.dataset_load_fn = dataset_load_fn

        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_dataset(self):
        return self.dataset_load_fn(self.dataset_name, self)