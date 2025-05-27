import torch

import argparse
import numpy as np

from DistributedSim.sim_builder import *
from DistributedSim.sim_config import *
from DistributedSim.strategy.strategy import *
from DistributedSim.strategy.diloco import *

from DistributedSim.models.nanogpt import GPT, GPTConfig
from DistributedSim.dataset.nanogpt.build_dataset import *

from nanogpt import arg_parse, config_gen, gen_gpt_config

def main():
    parser = arg_parse()

    parser.add_argument("--diloco_interval", type=int, default=100)
    parser.add_argument("--device_type", type=str, default="mps")
    parser.add_argument("--local_minibatch_size", type=int, default=4)
    parser.add_argument("--outer_lr", type=float, default=0.7)
    parser.add_argument("--nesterov", type=bool, default=True)
    parser.add_argument("--outer_momentum", type=float, default=0.9)

    args = parser.parse_args()

    gpt_config = gen_gpt_config(args)

    config = config_gen(args, gpt_config)

    config.strategy_class = DiLoCoStrategy
    config.strategy_config.diloco_interval = args.diloco_interval
    config.strategy_config.outer_optimizer_cls = torch.optim.SGD
    config.strategy_config.outer_optimizer_kwargs = {
        'lr': args.outer_lr,
        'nesterov': args.nesterov,
        'momentum': args.outer_momentum,
    }

    simbuilder = DistributedSimBuilder(config)

    simbuilder.execute()


if __name__ == "__main__":
    main()
