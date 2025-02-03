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
from DistributedSim.models.dataset_small import *


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
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--eval_interval", type=int, default=100)
    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load dataset from HuggingFace
    # train_data, val_data, args.vocab_size = get_dataset(args)
    train_data, val_data, args.vocab_size = get_dataset_small(args)
    print(f'Vocab size: {args.vocab_size}')

    # Create datasets
    train_dataset = GPTTrainDataset(train_data, args.block_size)
    val_dataset = GPTTrainDataset(val_data, args.block_size)

    gpt_config = GPTConfig(
        block_size=args.block_size,
        vocab_size=args.vocab_size,
        n_layer=2,
        n_head=2,
        n_embd=128,
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
        # val_size=256,
        val_size=64,
        gradient_class=SimpleReduceGradient,
        # gradient_class=SPARTAGradient,
        gradient_config=GradientConfig(
            optimizer_class=torch.optim.SGD,
            optimizer_kwargs={},
            p_sparta=0.005,
            async_sparta_delay=0,
            lr_scheduler='lambda_cosine',
            warmup_steps=1000,
            cosine_anneal=True,
            max_local_steps=3000,
        ),
        save_dir=args.checkpoint_dir,
        checkpoint_interval=100,
        wandb_project="nanogpt_char_cpu",
        device='mps',
        eval_interval=args.eval_interval,
        seed=args.seed,
        lr_scale=args.batch_size / 16,
    )

    simbuilder = LocalSimBuilder(config)

    simbuilder.execute()


if __name__ == "__main__":
    main()