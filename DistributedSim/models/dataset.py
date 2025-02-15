import torch
from datasets import load_dataset
import argparse
import numpy as np
import math
import os
import torch
from datasets import load_dataset
from datasets import Dataset, DatasetDict

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

def get_dataset(dataset, block_size=1024, char=False):
    """
    Loads and preprocesses the dataset with caching, using either a custom character-level tokenizer
    or the GPT2 tokenizer.
    
    Args:
        dataset: a string identifier ("shakespeare", "wikitext", "code", or "owt")
        block_size: the sequence block size.
        char (bool): If True, use character-level tokenization; otherwise, use GPT-2 tokenization.
    
    Returns:
        (train_data, val_data, vocab_size)
    """
    # Decide cache locations based on tokenization mode.
    if char:
        cache_dir = os.path.join("cache", f"{dataset}_char")
    else:
        cache_dir = os.path.join("cache", dataset)
    os.makedirs(cache_dir, exist_ok=True)
    data_cache_file = os.path.join(cache_dir, f"data_block{block_size}.pt")

    # Check for cached data.
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
        raw_dataset = load_dataset("Skylion007/openwebtext", trust_remote_code=True)
        # The dataset comes with a single split ("train"); extract the text field.
        all_texts = [item["text"] for item in raw_dataset["train"] if item["text"].strip()]
        # Create train/validation splits (90/10)
        split_idx = int(len(all_texts) * 0.9)
        train_texts = all_texts[:split_idx]
        test_texts = all_texts[split_idx:]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

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
        custom_tokenize = lambda text: {"input_ids": [char_int[c] for c in text["text"]]}
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
    tokenized_dataset = standardized_dataset.map(
        custom_tokenize,
        remove_columns=["text"]
    )

    # Convert tokenized lists to tensors with fixed block size
    def convert_to_tensor(examples):
        all_ids = []
        for ids in examples["input_ids"]:
            if len(ids) > 0:
                all_ids.extend(ids + [eos_token_id])
        num_blocks = len(all_ids) // block_size
        if len(all_ids) == 0 or num_blocks == 0:
            return {"input_ids": torch.tensor([])}
        all_ids = all_ids[: num_blocks * block_size]
        tensor_data = torch.tensor(all_ids).view(-1, block_size)
        return {"input_ids": tensor_data}

    print("Converting tokenized dataset to tensors...")
    tensor_dataset = tokenized_dataset.map(
        convert_to_tensor, batched=True, remove_columns=tokenized_dataset["train"].column_names
    )

    # Build train and validation tensors
    train_tensors = [torch.as_tensor(x) for x in tensor_dataset["train"]["input_ids"] if len(x) > 0]
    val_tensors = [torch.as_tensor(x) for x in tensor_dataset["test"]["input_ids"] if len(x) > 0]
    train_data = torch.cat(train_tensors, dim=0) if train_tensors else torch.tensor([])
    val_data = torch.cat(val_tensors, dim=0) if val_tensors else torch.tensor([])

    print(f"Train data size: {train_data.shape}, Val data size: {val_data.shape}")

    # Cache the processed dataset
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
        train_data, val_data, vocab_size, tokenizer_info = get_dataset(args.dataset, args.block_size, return_tokenizer=True, char=True)
        print(train_data.shape, val_data.shape, vocab_size)
        # For example, print a sample slice decoded using the char-level mapping.
        int_char = tokenizer_info["int_char"]
        sample_text = ''.join([int_char[x] for x in train_data[10000:10100].cpu().numpy()])
        print(sample_text)
    else:
        train_data, val_data, vocab_size, tokenizer = get_dataset(args.dataset, args.block_size, return_tokenizer=True, char=False)
        print(train_data.shape, val_data.shape, vocab_size)
        # Using GPT2's decode function to print a sample slice.
        sample_text = tokenizer.decode(train_data[10000:10100].cpu().numpy().tolist())
        print(sample_text)