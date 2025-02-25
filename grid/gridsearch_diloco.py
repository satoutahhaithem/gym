import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Grid search for lr and outer_lr with additional command line arguments.")
    parser.add_argument('--device_type', default="cuda", help="Type of device to use (default: cuda)")
    parser.add_argument('--devices', nargs='+', default=["0", "1", "2", "3"], help="List of device IDs (default: 0 1 2 3)")
    parser.add_argument('--dataset', default="wikitext", help="Dataset to use (default: wikitext)")
    parser.add_argument('--local_minibatch_size', default="4", help="Local minibatch size (default: 4)")
    args = parser.parse_args()

    # Define grid search values for lr and outer_lr.
    lr_values = [0.0001, 0.0002, 0.0003]
    outer_lr_values = [0.6, 0.7, 0.8]

    # Base command components that remain the same for all runs.
    base_command = [
        "python", "run/nanogpt_diloco.py",
        "--dataset", args.dataset,
        "--local_minibatch_size", args.local_minibatch_size,
        "--batch_size", "32",
        "--num_nodes", "8",
        "--diloco_interval", "100",
        "--model_size", "base",
        "--cosine_anneal",
        "--warmup_steps", "200",
        "--max_steps", "1000",
        "--wandb_project", "nanogpt_wikitext_n8_grid",
        "--device_type", args.device_type,
        "--block_size", "1024",
    ]
    
    # Add devices from command line.
    base_command.extend(["--devices"] + args.devices)

    # Loop over each combination of lr and outer_lr.
    for lr in lr_values:
        for outer_lr in outer_lr_values:
            # Create the full command by appending lr, outer_lr, and a unique wandb_name.
            command = base_command.copy()  # copy to avoid modifying the original list
            command.extend([
                "--lr", str(lr), 
                "--outer_lr", str(outer_lr),
                "--wandb_name", f'lr{lr}_outerlr_{outer_lr}'
            ])

            # Print the command being executed.
            print("Running command:", " ".join(command))

            # Execute the command. The call will block until the process finishes.
            subprocess.run(command)

if __name__ == "__main__":
    main()
