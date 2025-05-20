import torch

import argparse
import numpy as np

from DistributedSim.sim_builder import *
from DistributedSim.sim_config import *
from DistributedSim.gradient_strategy.gradient_strategy import *
from DistributedSim.gradient_strategy.demo_gradient import *

from DistributedSim.models.nanogpt import GPT, GPTConfig
from DistributedSim.dataset.nanogpt.build_dataset import *

from nanogpt import arg_parse, config_gen, gen_gpt_config

def main():
    parser = arg_parse()

    # Add DeMo-specific hyperparameters
    parser.add_argument("--compression_decay", type=float, default=0.999, 
                        help="Decay factor for gradient error feedback")
    parser.add_argument("--compression_topk", type=int, default=32, 
                        help="Number of top-k elements to keep in compression")
    parser.add_argument("--compression_chunk", type=int, default=64, 
                        help="Chunk size for DCT transformation")
    parser.add_argument("--weight_decay", type=float, default=0.0, 
                        help="Weight decay factor")

    args = parser.parse_args()

    gpt_config = gen_gpt_config(args)

    config = config_gen(args, gpt_config)

    config.gradient_class = DeMoGradient
    # Configure DeMo optimizer parameters
    config.gradient_config.optimizer_kwargs = {
        'lr': args.lr,

        'compression_decay': args.compression_decay,
        'compression_topk': args.compression_topk,
        'compression_chunk': args.compression_chunk,
        'weight_decay': args.weight_decay,
    }

    simbuilder = LocalSimBuilder(config)

    simbuilder.execute()

if __name__ == "__main__":
    main()