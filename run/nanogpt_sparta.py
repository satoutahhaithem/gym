from DistributedSim.trainer import LocalTrainer
from DistributedSim.strategy.sparta import SPARTAStrategy
from DistributedSim.example.nanogpt.nanogpt import GPT, GPTConfig
from DistributedSim.example.nanogpt.dataset import get_dataset
from DistributedSim.strategy.optim import OptimSpec

import torch
import argparse
import numpy as np

def gen_wandb_name(args):
    name = f"p{args.p_sparta}_n{args.num_nodes}_lr{args.lr:.0e}"
    return name

def arg_parse():
    # Command line arguments
    parser = argparse.ArgumentParser(conflict_handler='resolve')

    parser.add_argument(
        "--dataset", type=str, default="shakespeare", 
        help="which dataset to use (shakespeare, wikitext, code, owt)"
    )
    parser.add_argument("--start_pc", type=float, default=0.0)
    parser.add_argument("--end_pc", type=float, default=0.9)
    parser.add_argument("--val_start_pc", type=float, default=0.9)
    parser.add_argument("--val_end_pc", type=float, default=1.0)
    parser.add_argument("--block_size", type=int, default=1024)

    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
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

    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--val_size", type=int, default=256)

    # SPARTA-specific arguments
    parser.add_argument("--p_sparta", type=float, default=0.005)
    parser.add_argument("--async_sparta_delay", type=int, default=0)
    parser.add_argument("--schedule_p", action='store_true')
    parser.add_argument("--p_min_factor", type=float, default=0.1)
    parser.add_argument("--fault_rate", type=float, default=0.0)

    return parser

def main():
    parser = arg_parse()
    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

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
        schedule_p=args.schedule_p,
        p_min_factor=args.p_min_factor,
        fault_rate=args.fault_rate
    )

    # Train
    trainer.fit(
        num_epochs=args.epochs,
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