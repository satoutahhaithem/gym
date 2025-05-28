import argparse

def add_common_args(parser):
  """Add common arguments that are shared across all nanogpt scripts."""
  
  # Dataset arguments
  parser.add_argument(
    "--dataset", type=str, default="shakespeare", 
    help="which dataset to use (shakespeare, wikitext, code, owt)"
  )
  parser.add_argument("--start_pc", type=float, default=0.0)
  parser.add_argument("--end_pc", type=float, default=0.9)
  parser.add_argument("--val_start_pc", type=float, default=0.9)
  parser.add_argument("--val_end_pc", type=float, default=1.0)
  parser.add_argument("--block_size", type=int, default=1024)

  # Training arguments
  parser.add_argument("--num_nodes", type=int, default=1)
  parser.add_argument("--device", type=str, default="cpu")
  parser.add_argument("--epochs", type=int, default=1)
  parser.add_argument(
    "--model_size", type=str, default="small", choices=["small", "base", "medium", "large", "xl"]
  )

  # Optimization arguments
  parser.add_argument("--batch_size", type=int, default=16)
  parser.add_argument("--minibatch_size", type=int, default=None)
  parser.add_argument("--lr", type=float, default=0.001)
  parser.add_argument("--max_norm", type=float, default=1.0)
  parser.add_argument("--warmup_steps", type=int, default=1000)
  parser.add_argument("--max_steps", type=int, default=10000)
  parser.add_argument("--cosine_anneal", action='store_true')

  # Logging and reproducibility
  parser.add_argument("--seed", type=int, default=1337)
  parser.add_argument("--wandb_project", type=str, default=None)
  parser.add_argument("--wandb_name", type=str, default=None)
  parser.add_argument("--val_size", type=int, default=256)

  return parser

def get_common_parser():
  """Get a parser with common arguments."""
  parser = argparse.ArgumentParser(conflict_handler='resolve')
  return add_common_args(parser) 