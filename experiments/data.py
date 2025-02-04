import numpy as np
import torch
from torch.utils.data import Dataset
import math

class TextDataset(Dataset):
    def __init__(self, bin_file_path, dtype=np.int32, seq_length=1024, train=False):
        """
        Args:
            bin_file_path (str): Path to the .bin file.
            dtype (type): Data type of the tokenized data (default: np.int32).
            seq_length (int): The fixed length of each sequence (x).
        """
        self.bin_file_path = bin_file_path
        self.dtype = dtype
        self.seq_length = seq_length

        # Create a memmap object for the entire binary file
        self.data = np.memmap(self.bin_file_path, dtype=self.dtype, mode="r")
        if train:
            self.data = self.data[:int(len(self.data) * 0.9)]
        else:
            self.data = self.data[int(len(self.data) * 0.9):]

        # Compute the total number of tokens in the dataset
        self.num_tokens = len(self.data)

        # Calculate how many sequences we can extract given the context length
        self.num_sequences = math.floor(self.num_tokens / (self.seq_length + 1))

    def __len__(self):
        # Return the number of sequences available based on the fixed seq_length
        return self.num_sequences

    def __getitem__(self, idx):
        """
        Get the sequence at index idx.
        Returns the token IDs (x) and the next token (y) as a torch tensor.
        """
        start_idx = idx * (self.seq_length + 1)
        end_idx = start_idx + (self.seq_length + 1)
        sequence = self.data[start_idx:end_idx].astype(np.int32)

        x = torch.tensor(sequence[:-1], dtype=torch.long)  # Input sequence
        y = torch.tensor(sequence[1:], dtype=torch.long)

        return x, y