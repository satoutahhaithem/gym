import torch

import argparse
import numpy as np

from DistributedSim.sim_builder import *
from DistributedSim.sim_config import *
from DistributedSim.strategy.strategy import *

from DistributedSim.models.nanogpt import GPT, GPTConfig
from DistributedSim.dataset.nanogpt.build_dataset import *
from DistributedSim.dataset.nanogpt.dataset import get_dataset
from DistributedSim.dataset.dataset import DatasetConfig

def gen_wandb_name(args):
    name = f"bs{args.batch_size}_lr{args.lr:.0e}_warm{args.warmup_steps}_max{args.max_steps}"
    return name

def arg_parse():
    # Command line arguments
    parser = argparse.ArgumentParser(conflict_handler='resolve')

    parser.add_argument(
        "--dataset", type=str, default="shakespeare", 
        help="which dataset to use (shakespeare, wikitext, code, owt)"
    )
    parser.add_argument("--start_pc", type=float, default=0.0)
    parser.add_argument("--end_pc", type=float, default=1.0)
    parser.add_argument("--block_size", type=int, default=1024)

    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--device_type", type=str, default="")
    parser.add_argument("--devices", type=int, nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--model_size", type=str, default="small", choices=["small", "base", "medium", "large", "xl"]
    )

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--minibatch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--cosine_anneal", action='store_true')
    parser.add_argument("--autocast", action='store_true')

    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint_interval", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--val_size", type=int, default=256)
    parser.add_argument("--dataset_proportion", type=float, default=1.0)
    parser.add_argument("--val_proportion", type=float, default=0.1)
    parser.add_argument("--correlation_interval", type=int, default=None)

    return parser

def gen_gpt_config(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    gpt_config = GPTConfig.gpt2_size_map(args.model_size)

    return gpt_config

def config_gen(args, gpt_config):
    config = SimConfig(
        model_class=GPT,
        model_config=gpt_config,

        num_epochs=args.epochs,
        num_nodes=args.num_nodes,
        device_type=args.device_type,
        devices=args.devices,
        autocast=args.autocast,

        minibatch_size=args.minibatch_size if args.minibatch_size else args.batch_size,
        block_size=args.block_size,
        val_size=args.val_size,
        save_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        eval_interval=args.eval_interval,
        correlation_interval=args.correlation_interval,

        strategy_config=StrategyConfig(
            optimizer_class=torch.optim.AdamW,
            optimizer_kwargs={
                'lr': args.lr,
            },
            max_norm=args.max_norm,
            lr_scheduler='lambda_cosine',
            warmup_steps=args.warmup_steps,
            cosine_anneal=args.cosine_anneal,
            max_local_steps=args.max_steps,        
        ),

        dataset_config=DatasetConfig(
            dataset_load_fn=get_dataset,
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            device=args.device_type,
            block_size=args.block_size,
            dataset_proportion=args.dataset_proportion,
            val_proportion=args.val_proportion,
            gpt_config=gpt_config,
        ),

        seed=args.seed,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name if args.wandb_name else gen_wandb_name(args),
    )

    return config