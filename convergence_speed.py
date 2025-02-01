import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import string
from tqdm import tqdm
import re
import random
import os
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from simulator.sim_builder import *
from simulator.sim_config import *
from simulator.gradient_strategy import *
from simulator.demo import *

from nanogpt import *

## Config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

token_dict = {letter:i for letter,i in zip(string.ascii_lowercase, range(1,27))}
token_dict[' '] = 0
token_dict['.'] = 27

sentences_file = 'sentences.txt'

allowed_chars = set(string.ascii_lowercase + ' \n')

tt_split = 0.9
word_count_threshold = 1000

vocab_size = 28  # a-z and space, . chars
context_len = 64
d_model = 32
dim_ff = 64

batch_size = 64
epochs = 3

num_nodes = 4


def main():
    config = setup_config(trim_dataset=1000)
    config = setup_config(trim_dataset=None)

    topk_list = [1, 4, 16]

    # # Firstly, train with no DeMo
    # config.graident_class = SimpleReduceGradient
    # config.optimizer_class = torch.optim.SGD
    # config.optimizer_kwargs = {'lr': 0.01}

    # simbuilder = LocalSimBuilder(config)
    # simbuilder.execute()

    # Next, we train with DeMo
    train_losses = []
    val_losses = []
    val_accuracies = []

    for topk in topk_list:
        config.graident_class = DeMoGradient
        config.optimizer_class = None
        config.optimizer_kwargs = {
            'lr': 0.01,
            'compression_topk': topk,
        }

        simbuilder = LocalSimBuilder(config)
        train_loss_series, val_loss_series, val_accuracy_series = \
            simbuilder.execute()

        train_losses.append(train_loss_series)
        val_losses.append(val_loss_series)
        val_accuracies.append(val_accuracy_series)

    train_loss_df = pd.DataFrame({f'loss_{k}':x for k, x in zip(topk_list, train_losses)})
    val_loss_df = pd.DataFrame({f'loss_{k}':x for k, x in zip(topk_list, val_losses)})
    val_accuracy_df = pd.DataFrame({f'accuracy_{k}':x for k, x in zip(topk_list, val_accuracies)})
    
    train_loss_df.to_parquet('log/train_loss.pq')
    val_loss_df.to_parquet('log/val_loss.pq')
    val_accuracy_df.to_parquet('log/val_accuracy.pq')
    
if __name__ == '__main__':
    os.environ['VERBOSITY'] = '1'
    main()
