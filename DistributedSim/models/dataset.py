import torch
import argparse
import numpy as np
import math
import os
import torch
from datasets import load_dataset, Dataset, DatasetDict

def generate_char_vocab():
    """
    Generates a fixed character vocabulary and returns two mappings:
    char -> int, int -> char, and also the special end-of-sequence token id.
    """
    vocab = ' !$&\',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz\n'
    char_int = {char: i for i, char in enumerate(vocab)}
    int_char = {i: char for i, char in enumerate(vocab)}

    # Define a special end-of-sequence token.
    eos_token = '<EOS>'
    char_int[eos_token] = len(char_int)
    eos_token_id = char_int[eos_token]
    return char_int, eos_token_id

def get_dataset(dataset, block_size=1024, char=False, rank=None, world_size=None):
    """
    Loads and preprocesses the dataset with caching, using either a custom character-level tokenizer
    or the GPT2 tokenizer. If rank and world_size are provided, only loads the relevant shard.
    
    Args:
        dataset: a string identifier ("shakespeare", "wikitext", "code", or "owt")
        block_size: the sequence block size.
        char (bool): If True, use character-level tokenization; otherwise, use GPT-2 tokenization.
        rank (int, optional): The rank of the current process
        world_size (int, optional): Total number of processes
    """
    # Decide cache locations based on tokenization mode and rank
    if char:
        cache_dir = os.path.join("cache", f"{dataset}_char")
    else:
        cache_dir = os.path.join("cache", dataset)
    os.makedirs(cache_dir, exist_ok=True)
    
    # Add rank to cache file name if distributed
    rank_suffix = f"_rank{rank}_of{world_size}" if rank is not None else ""
    data_cache_file = os.path.join(cache_dir, f"data_block{block_size}_{rank_suffix}.pt")

    # Check for cached data
    if os.path.exists(data_cache_file):
        print(f"Loading cached dataset from {data_cache_file}")
        cached_data = torch.load(data_cache_file)
        return cached_data["train"], cached_data["val"], cached_data["vocab_size"]

    # Load the raw dataset and standardize to simple text format
    print(f"Loading dataset: {dataset} {'(char-level)' if char else '(GPT2 tokenization)'}")
    if dataset == "shakespeare":
        raw_dataset = load_dataset("Trelis/tiny-shakespeare")
        train_texts = [text["Text"] for text in raw_dataset["train"]]
        test_texts = [text["Text"] for text in raw_dataset["test"]]
    elif dataset == "wikitext":
        raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        train_texts = [text for text in raw_dataset["train"]["text"] if text.strip()]
        test_texts = [text for text in raw_dataset["test"]["text"] if text.strip()]
    elif dataset == "code":
        raw_dataset = load_dataset("codeparrot/codeparrot-clean-train", split="train[:1%]")
        train_texts = [text for text in raw_dataset["content"]]
        # Since this dataset doesn't come with a test split, create one
        split_idx = int(len(train_texts) * 0.9)
        test_texts = train_texts[split_idx:]
        train_texts = train_texts[:split_idx]
    elif dataset == "owt":
        # Load the OpenWebText dataset from Hugging Face
        raw_dataset = load_dataset("Skylion007/openwebtext", trust_remote_code=True, split='train[:10%]')
        # The dataset comes with a single split ("train"); extract the text field.
        all_texts = raw_dataset["text"]
        # Create train/validation splits (90/10)
        split_idx = int(len(all_texts) * 0.9)
        train_texts = all_texts[:split_idx]
        test_texts = all_texts[split_idx:]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    print(len(train_texts), len(test_texts))

    # Concatenate all texts with EOS token between them
    # train_text = "\n".join(train_texts)
    # test_text = "\n".join(test_texts)

    # Create standardized dataset format using datasets.Dataset
    standardized_dataset = DatasetDict({
        "train": Dataset.from_dict({"text": train_texts}),
        "test": Dataset.from_dict({"text": test_texts})
    })

    # Initialize the tokenizer
    if char:
        # Use character-based tokenization
        char_int, eos_token_id = generate_char_vocab()
        vocab_size = len(char_int)
        custom_tokenize = lambda text: {"input_ids": [[char_int[c] for c in t] for t in text["text"]]}
    else:
        # Use GPT2 tokenizer
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        vocab_size = tokenizer.vocab_size
        eos_token_id = tokenizer.eos_token_id
        custom_tokenize = lambda text: tokenizer(text["text"], truncation=True, max_length=block_size)

    print("Tokenizing dataset...")
    print(standardized_dataset)
    tokenized_dataset = standardized_dataset.map(
        custom_tokenize,
        remove_columns=["text"],
        batched=True,
        num_proc=os.cpu_count()
    )

    # Convert tokenized lists to tensors with fixed block size
    def convert_to_tensor(examples):
        # Flatten all ids and add EOS tokens
        all_ids = np.concatenate([np.array(ids + [eos_token_id]) for ids in examples["input_ids"] if ids])
        num_blocks = len(all_ids) // block_size
        if num_blocks == 0:
            return {"input_ids": torch.tensor([])}
        all_ids = all_ids[: num_blocks * block_size]
        tensor_data = torch.from_numpy(all_ids.reshape(-1, block_size))
        return {"input_ids": tensor_data}

    print("Converting tokenized dataset to tensors...")
    tensor_dataset = tokenized_dataset.map(
        convert_to_tensor,
        remove_columns=tokenized_dataset["train"].column_names,
        batched=True,
        num_proc=os.cpu_count()
    )

    # Build train and validation tensors
    tensor_dataset["train"].set_format("torch", columns=["input_ids"])
    tensor_dataset["test"].set_format("torch", columns=["input_ids"])
    train_data = tensor_dataset["train"]["input_ids"].reshape(-1)
    val_data = tensor_dataset["test"]["input_ids"].reshape(-1)

    print(f"Train data size: {train_data.shape}, Val data size: {val_data.shape}")

    # If rank is provided, only keep the relevant shard of the data
    if rank is not None and world_size is not None:
        # Calculate shard size and indices
        train_size = len(train_data)
        shard_size = train_size // world_size
        start_idx = rank * shard_size
        end_idx = start_idx + shard_size if rank < world_size - 1 else train_size
        
        # Only keep the relevant shard
        train_data = train_data[start_idx:end_idx]
        
        # For validation, we can either shard it or keep a small subset for each rank
        val_size = len(val_data)
        val_shard_size = val_size // world_size
        val_start_idx = rank * val_shard_size
        val_end_idx = val_start_idx + val_shard_size if rank < world_size - 1 else val_size
        val_data = val_data[val_start_idx:val_end_idx]

    # Cast to int32 before caching
    train_data = train_data.to(torch.int32)
    val_data = val_data.to(torch.int32)

    # Cache the processed dataset shard
    cache_data = {
        "train": train_data,
        "val": val_data,
        "vocab_size": vocab_size,
    }
    torch.save(cache_data, data_cache_file)

    return train_data, val_data, vocab_size

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--dataset", type=str, default="shakespeare",
                        help="Dataset: shakespeare, wikitext, or code")
    parser.add_argument("--char", action="store_true",
                        help="Enable character-level tokenization")
    args = parser.parse_args()

    if args.char:
        # When using character-level tokenization, get the char vocab info.
        train_data, val_data, vocab_size = get_dataset(args.dataset, args.block_size, char=True)
        print(train_data.shape, val_data.shape, vocab_size)
    else:
        train_data, val_data, vocab_size = get_dataset(args.dataset, args.block_size, char=False)
        print(train_data.shape, val_data.shape, vocab_size)
