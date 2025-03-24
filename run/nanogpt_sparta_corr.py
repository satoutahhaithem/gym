import torch

import argparse
import numpy as np

from DistributedSim.sim_builder import *
from DistributedSim.sim_config import *
from DistributedSim.gradient_strategy.gradient_strategy import *
from DistributedSim.gradient_strategy.sparta_corr_gradient import *

from DistributedSim.models.nanogpt import GPT, GPTConfig
from DistributedSim.dataset.build_dataset import *

from nanogpt import arg_parse, config_gen, gen_gpt_config

def gen_wandb_name(args):
    name = f"p{args.p_sparta}_n{args.num_nodes}_lr{args.learning_rate:.0e}"
    return name

def main():
    parser = arg_parse()

    parser.add_argument("--p_sparta", type=float, default=0.005)
    parser.add_argument("--target_corr", type=float, default=0.8)
    parser.add_argument("--rebalance_frequency", type=int, default=10)
    parser.add_argument("--p_lr", type=int, default=0.01)

    args = parser.parse_args()

    gpt_config = gen_gpt_config(args)

    config = config_gen(args, gpt_config)

    config.gradient_class = SPARTACorrelationGradient
    config.gradient_config.p_sparta = args.p_sparta
    config.gradient_config.target_corr = args.target_corr
    config.gradient_config.rebalance_frequency = args.rebalance_frequency
    config.gradient_config.p_lr = args.p_lr

    simbuilder = LocalSimBuilder(config)

    simbuilder.execute()


if __name__ == "__main__":
    main()