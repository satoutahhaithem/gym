import argparse
import random
import subprocess

def random_search(args):
    while True:
        # Define ranges for hyperparameters
        batch_size = random.choice([8, 16])
        learning_rate = random.uniform(5e-5, 1e-3)
        warmup_steps = random.randint(100, 400)
        max_steps = random.randint(2000, 5000)

        if args.char_dataset:
            max_steps *= 3

        # Construct the command
        command = [
            "python", "experiments/nanogpt_gpu.py",
            "--gpu_offset", str(args.gpu_offset),
            '--dataset', args.dataset,
            '--wandb_project', f'grid_{args.dataset}',
            '--model_size', args.model_size,

            "--batch_size", str(batch_size),
            "--learning_rate", str(learning_rate),
            "--warmup_steps", str(warmup_steps),
            "--max_steps", str(max_steps),
        ]

        if args.char_dataset:
            command.append('--char_dataset')

        # Print the command for debugging
        print("Running command:", " ".join(command))

        # Run the command
        result = subprocess.run(command)

        # Check if the command was successful
        if result.returncode != 0:
            print("Command failed with return code:", result.returncode)
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random search over hyperparameters for nanogpt_gpu_sparts.py")
    parser.add_argument("--gpu_offset", type=int, required=True, help="GPU offset to use")
    parser.add_argument(
        "--dataset", type=str, default="shakespeare", help="which dataset to use (shakespeare, wikitext, code)"
    )
    parser.add_argument(
        "--model_size", type=str, default="small", choices=["small", "base", "medium", "large", "xl"]
    )
    parser.add_argument('--char_dataset', action='store_true')

    args = parser.parse_args()

    random_search(args)