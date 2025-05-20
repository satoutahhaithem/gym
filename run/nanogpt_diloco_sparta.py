import torch

import argparse
import numpy as np

from DistributedSim.sim_builder import *
from DistributedSim.sim_config import *
from DistributedSim.gradient_strategy.gradient_strategy import *
from DistributedSim.gradient_strategy.diloco_sparta_gradient import *

from DistributedSim.models.nanogpt import GPT, GPTConfig
from DistributedSim.dataset.nanogpt.build_dataset import *

from nanogpt import arg_parse, config_gen, gen_gpt_config

def main():
    parser = arg_parse()

    parser.add_argument("--diloco_interval", type=int, default=100)
    parser.add_argument('--outer_lr', type=float, default=0.7)
    parser.add_argument("--nesterov", type=bool, default=True)
    parser.add_argument("--outer_momentum", type=float, default=0.9)

    parser.add_argument("--p_sparta", type=float, default=0.005)
    parser.add_argument("--sparta_interval", type=int, default=1)

    args = parser.parse_args()

    gpt_config = gen_gpt_config(args)

    config = config_gen(args, gpt_config)

    config.gradient_class = DiLoCoSPARTAGradient
    config.gradient_config.diloco_interval = args.diloco_interval
    config.gradient_config.outer_optimizer_cls = torch.optim.SGD
    config.gradient_config.outer_optimizer_kwargs = {
        'lr': args.outer_lr,
        'nesterov': args.nesterov,
        'momentum': args.outer_momentum,
    }

    config.gradient_config.p_sparta = args.p_sparta
    config.gradient_config.sparta_interval = args.sparta_interval

    simbuilder = LocalSimBuilder(config)

    simbuilder.execute()

if __name__ == "__main__":
    main()