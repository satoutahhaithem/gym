import torch

import argparse
import numpy as np

from DistributedSim.sim_builder import *
from DistributedSim.sim_config import *
from DistributedSim.gradient_strategy import *
from DistributedSim.demo import *

from DistributedSim.models.nanogpt import GPT, GPTConfig, GPTTrainDataset
from DistributedSim.models.dataset import *

from nanogpt import arg_parse, config_gen, gen_gpt_config

def main():
    parser = arg_parse()

    parser.add_argument("--diloco_interval", type=int, default=100)
    parser.add_argument('--outer_lr', type=float, default=0.7)

    args = parser.parse_args()

    train_data, val_data, args.vocab_size = get_dataset(args.dataset, 
                                                        block_size=args.block_size, 
                                                        char=args.char_dataset)

    train_dataset = GPTTrainDataset(train_data, args.block_size)
    val_dataset = GPTTrainDataset(val_data, args.block_size)

    gpt_config = gen_gpt_config(args)

    config = config_gen(args, train_dataset, val_dataset, gpt_config)

    config.gradient_class = DiLoCoGradient
    config.gradient_config.diloco_interval = args.diloco_interval
    config.gradient_config.outer_optimizer_cls = torch.optim.SGD
    config.gradient_config.outer_optimizer_kwargs = {
        'lr': args.outer_lr,
        'nesterov': True,
        'momentum': 0.9,
    }

    simbuilder = LocalSimBuilder(config)

    simbuilder.execute()

if __name__ == "__main__":
    main()