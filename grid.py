import argparse
import random
import subprocess

def random_search(gpu_offset):
    while True:
        # Define ranges for hyperparameters
        batch_size = random.choice([4, 8, 16, 32])
        learning_rate = random.uniform(5e-5, 1e-3)
        warmup_steps = random.randint(300, 2000)
        max_steps = random.randint(5000, 10000)

        # Construct the command
        command = [
            "python", "experiments/nanogpt_gpu_sparta.py",
            "--gpu_offset", str(gpu_offset),
            "--batch_size", str(batch_size),
            "--learning_rate", str(learning_rate),
            "--warmup_steps", str(warmup_steps),
            "--max_steps", str(max_steps),
            # "--max_steps", '10',
            '--dataset', 'wikitext'
        ]

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
    args = parser.parse_args()

    random_search(args.gpu_offset)