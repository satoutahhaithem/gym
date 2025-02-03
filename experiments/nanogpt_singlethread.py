import torch
import torch.nn as nn
from torch.nn import functional as F

import argparse
import numpy as np

from SingleThreadSim.sim_builder import *
from SingleThreadSim.sim_config import *
from SingleThreadSim.gradient_strategy import *
from SingleThreadSim.demo import *

from DistributedSim.models.nanogpt import *

from SingleThreadSim.dataset_small import *

def main():
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="shakespeare", help="which dataset to use (shakespeare, wikitext, code)"
    )
    parser.add_argument("--num_nodes", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=6e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-1)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--test_size", action='store_true')
    parser.add_argument("--eval_interval", type=int, default=100)
    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load dataset from HuggingFace
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
        gradient_class=FakeAllReduceGradient,
        # gradient_class=NoCommunicationGradient,
        gradient_config=GradientConfig(
            optimizer_class=torch.optim.SGD,
            optimizer_kwargs={
                'lr': args.learning_rate,
                # 'weight_decay': args.weight_decay,
                # 'betas': (args.beta1, args.beta2),
            },
            # lr_scheduler=torch.optim.lr_scheduler.StepLR,
            # lr_scheduler_kwargs={
            #     'step_size': 10,
            #     'gamma': 0.95
            # }
        ),
        save_dir=args.checkpoint_dir,
        wandb_project="nangpt_singlethread",
        checkpoint_interval=100,
        eval_interval=args.eval_interval,
        device='mps',
    )

    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir + '/' + config.wandb_project, exist_ok=True)

    simbuilder = SingleThreadSimBuilder(config)

    simbuilder.execute()


if __name__ == "__main__":
    main()