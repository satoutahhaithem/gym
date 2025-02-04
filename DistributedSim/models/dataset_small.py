import torch
from datasets import load_dataset
import argparse

def generate_char_vocab():
    vocab = ' !$&\',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz\n'
    char_int = {char: i for i, char in enumerate(vocab)}
    int_char = {i: char for i, char in enumerate(vocab)}

    # Define a special end-of-sequence token
    eos_token = '<EOS>'
    char_int[eos_token] = len(char_int)
    eos_token_id = char_int[eos_token]
    return char_int, int_char, eos_token_id

def get_dataset_small(args):
    char_int, int_char, eos_token_id = generate_char_vocab()

    print("Loading dataset: shakespeare")

    dataset = load_dataset("Trelis/tiny-shakespeare")

    print(f"Dataset loaded. Structure: {dataset}")

    def tokenize_function(examples):
        print(examples['Text'])
        return {"input_ids": [char_int[char] for char in examples["Text"]]}

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize_function, remove_columns=dataset["train"].column_names)

    # Convert to torch tensors
    def convert_to_tensor(examples):
        all_ids = []
        for ids in examples["input_ids"]:
            if len(ids) > 0:  # Skip empty sequences
                all_ids.extend(ids + [eos_token_id])

        if len(all_ids) == 0:
            return {"input_ids": torch.tensor([])}

        # Make sure we have complete blocks
        num_blocks = len(all_ids) // args.block_size
        if num_blocks == 0:
            return {"input_ids": torch.tensor([])}

        all_ids = all_ids[: num_blocks * args.block_size]
        tensor_data = torch.tensor(all_ids).view(-1, args.block_size)
        return {"input_ids": tensor_data}

    print("Converting to tensors...")
    tensor_dataset = tokenized_dataset.map(
        convert_to_tensor, batched=True, remove_columns=tokenized_dataset["train"].column_names
    )

    # Extract and concatenate all tensors
    train_data = torch.cat([torch.as_tensor(x) for x in tensor_dataset["train"]["input_ids"] if len(x) > 0], dim=0)
    val_data = torch.cat([torch.as_tensor(x) for x in tensor_dataset["test"]["input_ids"] if len(x) > 0], dim=0)

    print(f"Train data size: {train_data.shape}, Val data size: {val_data.shape}")
    return train_data, val_data, len(char_int)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--block_size", type=int, default=1024)
    args = args.parse_args()

    train_data, val_data, vocab_size = get_dataset_small(args)
    print(train_data.shape, val_data.shape, vocab_size)

    _, int_char, _ = generate_char_vocab()

    print(''.join([int_char[x] for x in train_data[10000:10100].cpu().numpy()]))
