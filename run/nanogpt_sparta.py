from exogym.trainer import LocalTrainer
from exogym.strategy.sparta import SPARTAStrategy
from exogym.example.nanogpt.nanogpt import GPT, GPTConfig
from exogym.example.nanogpt.dataset import get_dataset
from exogym.strategy.optim import OptimSpec
from common_args import get_common_parser

import torch
import numpy as np

def gen_wandb_name(args):
    name = f"p{args.p_sparta}_n{args.num_nodes}_lr{args.lr:.0e}"
    return name

def arg_parse():
    parser = get_common_parser()

    # SPARTA-specific arguments
    parser.add_argument("--p_sparta", type=float, default=0.005)
    parser.add_argument("--async_sparta_delay", type=int, default=0)

    return parser

def main():
    parser = arg_parse()
    args = parser.parse_args()

    # Get datasets
    train_dataset, vocab_size = get_dataset(
        args.dataset, 
        block_size=args.block_size, 
        device='cpu', 
        start_pc=args.start_pc, 
        end_pc=args.end_pc
    )
    val_dataset, vocab_size = get_dataset(
        args.dataset, 
        block_size=args.block_size, 
        device='cpu', 
        start_pc=args.val_start_pc, 
        end_pc=args.val_end_pc
    )

    # Create model
    gpt_config = GPTConfig.gpt2_size_map(args.model_size)
    gpt_config.vocab_size = vocab_size
    model = GPT(gpt_config)

    # Create trainer
    trainer = LocalTrainer(
        model, 
        train_dataset, 
        val_dataset, 
    )

    # Create optimizer spec
    optim = OptimSpec(
        torch.optim.AdamW,
        lr=args.lr
    )

    # Create SPARTA strategy
    strategy = SPARTAStrategy(
        optim_spec=optim,
        lr_scheduler='lambda_cosine',
        lr_scheduler_kwargs={
            'warmup_steps': args.warmup_steps,
            'cosine_anneal': args.cosine_anneal
        },
        max_norm=args.max_norm,
        p_sparta=args.p_sparta,
        async_sparta_delay=args.async_sparta_delay,
    )

    # Train
    trainer.fit(
        num_epochs=args.epochs,
        max_steps=args.max_steps,
        strategy=strategy,
        num_nodes=args.num_nodes,
        device=args.device,
        batch_size=args.batch_size,
        minibatch_size=args.minibatch_size or args.batch_size,
        val_size=args.val_size,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name or gen_wandb_name(args)
    )

if __name__ == "__main__":
    main()