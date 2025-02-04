import torch
import torch.nn as nn
from torch.nn import functional as F

import argparse
import numpy as np

from DistributedSim.sim_builder import *
from DistributedSim.sim_config import *
from DistributedSim.gradient_strategy import *
from DistributedSim.demo import *

from DistributedSim.models.nanogpt import *


def main():
    # Command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset", type=str, default="shakespeare", help="which dataset to use (shakespeare, wikitext, code)"
    )
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--test_size", action='store_true')
    parser.add_argument("--gpu_offset", type=int, default=0)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--wandb_project", type=str, default="nanogpt_small")
    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cudnn.allow_tf32 = True

    # Load dataset from HuggingFace
    train_data, val_data, args.vocab_size = get_dataset(args)
    print(f'Vocab size: {args.vocab_size}')

    # Create datasets
    train_dataset = GPTTrainDataset(train_data, args.block_size)
    val_dataset = GPTTrainDataset(val_data, args.block_size)

    if args.test_size:
        gpt_config = GPTConfig(
            block_size=args.block_size,
            vocab_size=args.vocab_size,
            n_layer=4,
            n_head=4,
            n_embd=256,
        )
    else:
        gpt_config = GPTConfig(
            block_size=args.block_size,
            vocab_size=args.vocab_size,
            n_layer=12,
            n_head=12,
            n_embd=768,
        )


    config = SimConfig(
        model_class=GPT,
        gpt_config=gpt_config,
        criterion_class=torch.nn.CrossEntropyLoss,
        num_epochs=args.epochs,
        num_nodes=args.num_nodes,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        val_size=256,
        gradient_class=SimpleReduceGradient,
        # gradient_class=SPARTAGradient,
        gradient_config=GradientConfig(
            optimizer_class=torch.optim.Adam,
            optimizer_kwargs={
                'lr': args.learning_rate,
                # 'weight_decay': args.weight_decay,
                # 'betas': (args.beta1, args.beta2),
            },
            # p_sparta=0.005,
            # async_sparta_delay=0,
            # lr_scheduler='lambda_cosine',
            # warmup_steps=300,
            # cosine_anneal=True,
            max_local_steps=3000,
            # lr_scheduler=torch.optim.lr_scheduler.StepLR,
            # lr_scheduler_kwargs={
            #     'step_size': 10,
            #     'gamma': 0.95
            # }
        ),
        save_dir=args.checkpoint_dir,
        checkpoint_interval=1000,
        wandb_project=args.wandb_project,
        device='cuda',
        gpu_offset=args.gpu_offset,
        eval_interval=args.eval_interval,
        lr_scale=1.0,
        seed=args.seed,
    )

    simbuilder = LocalSimBuilder(config)

    simbuilder.execute()


if __name__ == "__main__":
    main()