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
tt_split = 0.9


## Classes

class TextDataset(Dataset):
    def __init__(self, sentences_verb):
        self.data = torch.tensor([token_dict[x] for x in sentences_verb], device=device)

    def __len__(self):
        return self.data.shape[0] - context_len

    def __getitem__(self, i):
        return self.data[i:i+context_len], self.data[i+context_len]

class TransformerModel(nn.Module):
    def __init__(
        self, 
        vocab_size=None, 
        context_len=None, 
        d_model=128, 
        n_heads=4, 
        num_layers=2, 
        dim_ff=256, 
        dropout=0.1
    ):
        super(TransformerModel, self).__init__()

        self.vocab_size = vocab_size
        self.context_len = context_len
        self.d_model = d_model

        # Token and positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_len, d_model)

        # Transformer encoder layers
        self.encoder_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=dim_ff,
                dropout=dropout
            ),
            num_layers=num_layers
        )

        # Output layer to project to vocabulary size
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: (batch_size, context_len)
        batch_size, seq_len = x.size()

        # Create token and positional embeddings
        token_embeds = self.token_embedding(x)  # (batch_size, seq_len, d_model)
        pos_indices = torch.arange(seq_len, device=x.device).unsqueeze(0)  # (1, seq_len)
        pos_embeds = self.position_embedding(pos_indices)  # (1, seq_len, d_model)

        # Combine token and positional embeddings
        embeddings = token_embeds + pos_embeds  # (batch_size, seq_len, d_model)

        # Apply Transformer encoder (transpose for sequence-first format)
        embeddings = embeddings.transpose(0, 1)  # (seq_len, batch_size, d_model)
        encoded = self.encoder_layers(embeddings)  # (seq_len, batch_size, d_model)
        encoded = encoded.transpose(0, 1)  # (batch_size, seq_len, d_model)

        # Project final hidden states to vocabulary size
        logits = self.output_layer(encoded)  # (batch_size, seq_len, vocab_size)
        return logits[:, -1, :]

# Utils

def build_dataset(sentences_file, trim_dataset=None):
    print('Importing sentences...')

    sentences = []
    with open(sentences_file, 'r') as f:
        sentences = f.readlines()
    sentences = [x.replace('\'', '') for x in sentences]
#     sentences = [re.sub('\W+', ' ', x.lower()).strip() for x in sentences]
    sentences = [re.sub(r'\W+', ' ', x.lower()).strip() for x in sentences]

    sentences = list(set(sentences))


    map_dict = {x[0]:y for x,y in 
            pd.DataFrame(list(' '.join(sentences).split(' ')))\
                .value_counts().to_dict().items()}

    common_sentences = []

    for sentence in sentences:
        good = True

        for word in sentence.split(' '):
            if map_dict[word] < word_count_threshold:
                good = False
                break

        for char in sentence:
            if char not in allowed_chars:
                good = False
                break
        
        if good:
            common_sentences.append(sentence + '\n')

    random.shuffle(common_sentences)

    sentence_verb = '. '.join([x.strip() for x in common_sentences])
    print(trim_dataset)
    if trim_dataset:
        sentence_verb = sentence_verb[:trim_dataset]
        print(f'trimming dataset to length {len(sentence_verb)}')

    print(f'dataset of {len(common_sentences)} sentences')

    return TextDataset(sentence_verb[:int(tt_split * len(sentence_verb))]), \
        TextDataset(sentence_verb[int(tt_split * len(sentence_verb)):])

def setup_config(trim_dataset=None):
    config = SimConfig()

    config.train_dataset, config.val_dataset = build_dataset(sentences_file, trim_dataset=trim_dataset)

    config.model_class = TransformerModel
    config.model_kwargs = {
        'vocab_size':vocab_size,
        'context_len':context_len,
        'd_model':d_model,
        'dim_ff':dim_ff,
    }

    # config.gradient_class = SimpleReduceGradient

    # config.optimizer_class = torch.optim.Adam
    # config.optimizer_kwargs = {
    #         'lr': 0.01
    # }

    config.gradient_class = DeMoGradient

    config.criterion_class = torch.nn.CrossEntropyLoss


    config.num_epochs = epochs

    config.num_nodes = num_nodes

    return config

def main():
    config = setup_config(trim_dataset=None)

    config.optimizer_kwargs = {
        'lr': 0.01,
        'compression_topk': 32,
    }
    simbuilder = LocalSimBuilder(config)

    simbuilder.execute()

if __name__ == '__main__':
    os.environ['VERBOSITY'] = '1'
    main()
