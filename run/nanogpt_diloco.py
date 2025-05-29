from DistributedSim.trainer import LocalTrainer
from DistributedSim.strategy.diloco import DiLoCoStrategy
from DistributedSim.example.nanogpt.nanogpt import GPT, GPTConfig
from DistributedSim.example.nanogpt.dataset import get_dataset
from DistributedSim.strategy.optim import OptimSpec
from common_args import get_common_parser

import torch
import numpy as np

def gen_wandb_name(args):
  name = f"bs{args.batch_size}_lr{args.lr:.0e}_outer{args.outer_lr:.0e}_H{args.diloco_interval}"
  return name

def arg_parse():
  parser = get_common_parser()
  
  # DiLoCo-specific arguments
  parser.add_argument("--diloco_interval", type=int, default=100)
  parser.add_argument('--outer_lr', type=float, default=0.7)
  parser.add_argument("--nesterov", type=bool, default=True)
  parser.add_argument("--outer_momentum", type=float, default=0.9)

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
  inner_optim = OptimSpec(
    torch.optim.AdamW,
    lr=args.lr
  )

  # Create outer optimizer spec
  outer_optim = OptimSpec(
    torch.optim.SGD,
    lr=args.outer_lr,
    nesterov=args.nesterov,
    momentum=args.outer_momentum
  )

  # Create DiLoCo strategy
  strategy = DiLoCoStrategy(
    inner_optim_spec=inner_optim,
    outer_optim_spec=outer_optim,
    H=args.diloco_interval,
    lr_scheduler='lambda_cosine',
    lr_scheduler_kwargs={
      'warmup_steps': args.warmup_steps,
      'cosine_anneal': args.cosine_anneal
    },
    max_norm=args.max_norm
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