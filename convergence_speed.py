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

from sim_builder import *
from sim_config import *
from gradient_strategy import *
from demo import *

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
epochs = 10

num_nodes = 4


def main():
    config = setup_config()

    # # Firstly, train with no DeMo
    # config.graident_class = SimpleReduceGradient
    # config.optimizer_class = torch.optim.SGD
    # config.optimizer_kwargs = {'lr': 0.01}

    # simbuilder = SimBuilder(config)
    # simbuilder.execute()

    # Next, we train with DeMo
    for topk in [8, 16, 32, 64, 128]:
        config.graident_class = DeMoGradient
        config.optimizer_class = None
        config.optimizer_kwargs = {
            'lr': 0.01,
            'compression_topk': topk,
        }

        simbuilder = SimBuilder(config)
        simbuilder.execute()

        
    
if __name__ == '__main__':
    os.environ['VERBOSITY'] = '1'
    main()
