def grid_search_p_sparta():
    """
    Performs a grid search over p_sparta.
    Fixed parameters:
      - learning_rate is set to 0.001
      - batch_size is fixed to 8
      - runs on 2 nodes (via --num_nodes)
    Additional parameters are passed directly to the experiments/nanogpt.py script.
    """
    import argparse
    import subprocess

    parser = argparse.ArgumentParser(
        description="Grid search over p_sparta for nanogpt.py (learning_rate fixed to 0.001, batch_size=8, 2 nodes)."
    )
    parser.add_argument("--gpu_offset", type=int, required=True, help="GPU offset to use")
    parser.add_argument("--dataset", type=str, default="wikitext", help="Dataset to use")
    parser.add_argument("--wandb_project", type=str, default="wikitext_small", help="Wandb project name")
    parser.add_argument("--model_size", type=str, default="small", choices=["small", "base", "medium", "large", "xl"])
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of warmup steps")
    parser.add_argument("--max_steps", type=int, default=5000, help="Number of max steps")
    # Fixed to two nodes for this experiment.
    parser.add_argument("--num_nodes", type=int, default=2, help="Number of nodes")
    # Option to supply a comma-separated list of p_sparta values
    parser.add_argument(
        "--p_sparta_list",
        type=str,
        default="1,0.1,0.01,0.005,0.02",
        help="Comma-separated list of p_sparta values to grid search over",
    )
    # Parse known plus any extra arguments.
    args, unknown = parser.parse_known_args()

    # Convert the p_sparta list string to a list of float values.
    p_sparta_vals = [float(x.strip()) for x in args.p_sparta_list.split(",") if x.strip()]

    for p_val in p_sparta_vals:
        command = [
            "python", "experiments/nanogpt_sparta.py",
            "--gpu_offset", str(args.gpu_offset),
            "--dataset", args.dataset,
            "--wandb_project", args.wandb_project,
            "--model_size", args.model_size,
            "--batch_size", "8",  # fixed batch size
            "--learning_rate", "0.0005",  # fixed learning rate
            "--warmup_steps", str(args.warmup_steps),
            "--max_steps", str(args.max_steps),
            "--num_nodes", str(args.num_nodes),
            "--p_sparta", str(p_val),
            '--wandb_name', f'lr{0.0005}_p{p_val}'
        ]
        # Append any extra parameters.
        command.extend(unknown)
        print("Running command:", " ".join(command))
        result = subprocess.run(command)
        if result.returncode != 0:
            print("Command failed with return code", result.returncode)
            break

if __name__ == "__main__":
    grid_search_p_sparta()