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
from DistributedSim.models.dataset import *

from torch.profiler import profile, record_function, ProfilerActivity

def gen_wandb_name(args):
    name = f"p{args.p_sparta}_n{args.num_nodes}_lr{args.learning_rate:.0e}"
    return name

def main():
    # Command line arguments
    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--dataset", type=str, default="shakespeare", help="which dataset to use (shakespeare, wikitext, code, owt)"
    )
    parser.add_argument("--num_nodes", type=int, default=2)
    parser.add_argument("--cpu", action='store_true')
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--gpu_offset", type=int, default=0)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--wandb_project", type=str, default="nanogpt_small")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--p_sparta", type=float, default=0.005)
    parser.add_argument(
        "--model_size", type=str, default="small", choices=["small", "base", "medium", "large", "xl"]
    )
    parser.add_argument('--char_dataset', action='store_true')

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=10000)



    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cudnn.allow_tf32 = True

    if args.dataset == 'owt':
        train_dataset = TextDataset('../diloco-sim/examples/data/owt/openwebtext.bin', 
                                    dtype=np.uint16, train=True)
        val_dataset = TextDataset('../diloco-sim/examples/data/owt/openwebtext.bin', 
                                    dtype=np.uint16, train=False)
        args.vocab_size = 50304
    else:
        train_data, val_data, args.vocab_size = get_dataset(args, 
                                                        char=args.char_dataset)

        # Create datasets
        train_dataset = GPTTrainDataset(train_data, args.block_size)
        val_dataset = GPTTrainDataset(val_data, args.block_size)
    
    print(f'Vocab size: {args.vocab_size}')

    gpt_config = {
        "small": GPTConfig.gpt2_small,
        "base": GPTConfig.gpt2_base,
        "medium": GPTConfig.gpt2_medium,
        "large": GPTConfig.gpt2_large,
        "xl": GPTConfig.gpt2_xl,
    }[args.model_size]()
    gpt_config.vocab_size = args.vocab_size

    config = SimConfig(
        model_class=GPT,
        gpt_config=gpt_config,
        criterion_class=torch.nn.CrossEntropyLoss,
        num_epochs=args.epochs,
        num_nodes=args.num_nodes,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        dataset_name=f'{args.dataset}_char' if args.char_dataset else args.dataset,
        batch_size=args.batch_size,
        val_size=256,
        gradient_class=SPARTAGradient,
        gradient_config=GradientConfig(
            optimizer_class=torch.optim.Adam,
            optimizer_kwargs={
                'lr': args.learning_rate,
            },
            p_sparta=args.p_sparta,
            # async_sparta_delay=0,
            lr_scheduler='lambda_cosine',
            warmup_steps=args.warmup_steps,
            cosine_anneal=True,
            max_local_steps=args.max_steps,
            # lr_scheduler=torch.optim.lr_scheduler.StepLR,
            # lr_scheduler_kwargs={
            #     'step_size': 10,
            #     'gamma': 0.95
            # }
        ),
        save_dir=args.checkpoint_dir,
        checkpoint_interval=1000,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_name if args.wandb_name else gen_wandb_name(args),
         # device='cuda',
        device='cuda' if not args.cpu else 'cpu',
        gpu_offset=args.gpu_offset,
        eval_interval=args.eval_interval,
        lr_scale=1.0,
        seed=args.seed,
    )

    simbuilder = LocalSimBuilder(config)

    simbuilder.execute()


if __name__ == "__main__":
    main()