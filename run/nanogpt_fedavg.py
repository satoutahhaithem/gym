import argparse
import numpy as np

from DistributedSim.sim_builder import *
from DistributedSim.sim_config import *
from DistributedSim.gradient_strategy.gradient_strategy import *
from DistributedSim.gradient_strategy.federated_averaging import *

from DistributedSim.models.nanogpt import GPT, GPTConfig
from DistributedSim.dataset.nanogpt.build_dataset import *

from nanogpt import arg_parse, config_gen, gen_gpt_config

def main():
    parser = arg_parse()

    parser.add_argument("--H", type=int, default=100)
    parser.add_argument("--island_size", type=int, default=None)

    args = parser.parse_args()

    if args.island_size is None:
        args.island_size = args.num_nodes

    gpt_config = gen_gpt_config(args)

    config = config_gen(args, gpt_config)

    config.gradient_class = FedAvgGradient
    config.gradient_config.H = args.H
    config.gradient_config.island_size = args.island_size

    simbuilder = LocalSimBuilder(config)

    simbuilder.execute()

if __name__ == "__main__":
    main()