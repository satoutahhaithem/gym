def grid_search_lr():
    """
    Performs a grid search over learning rates.
    Fixed parameters:
      - batch_size is set to 8
      - runs on a single node
    Additional parameters are passed directly to the experiments/nanogpt.py script.
    """
    import argparse
    import subprocess

    parser = argparse.ArgumentParser(
        description="Grid search over learning rate for nanogpt.py (batch size fixed to 8, single node)."
    )
    # Required and default arguments for the base run
    parser.add_argument("--gpu_offset", type=int, required=True, help="GPU offset to use")
    parser.add_argument("--dataset", type=str, default="wikitext", help="Dataset to use")
    parser.add_argument("--wandb_project", type=str, default="wikitext_small", help="Wandb project name")
    parser.add_argument("--model_size", type=str, default="small", choices=["small", "base", "medium", "large", "xl"])
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of warmup steps")
    parser.add_argument("--max_steps", type=int, default=5000, help="Number of max steps")
    # Option to supply a comma-separated list of learning rates
    parser.add_argument(
        "--lr_list",
        type=str,
        default="0.0001,0.0005,0.001,0.005",
        help="Comma-separated list of learning rates to grid search over",
    )
    # Parse known arguments; other extra parameters are passed through.
    args, unknown = parser.parse_known_args()

    # Convert the comma-separated string of learning rates to a list of floats.
    lr_list = [float(x.strip()) for x in args.lr_list.split(",") if x.strip()]

    for lr in lr_list:
        command = [
            "python", "experiments/nanogpt.py",
            "--gpu_offset", str(args.gpu_offset),
            "--dataset", args.dataset,
            "--wandb_project", args.wandb_project,
            "--model_size", args.model_size,
            "--batch_size", "8",  # fixed batch size for this grid search
            "--learning_rate", str(lr),
            "--warmup_steps", str(args.warmup_steps),
            "--max_steps", str(args.max_steps),
        ]
        # Append any extra command-line arguments.
        command.extend(unknown)
        print("Running command:", " ".join(command))
        result = subprocess.run(command)
        if result.returncode != 0:
            print("Command failed with return code", result.returncode)
            break

if __name__ == "__main__":
    grid_search_lr()