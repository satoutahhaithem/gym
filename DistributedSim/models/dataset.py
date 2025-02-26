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

    assert rank is not None
    
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
        raw_dataset = raw_dataset.map(
            lambda x: {'text': x['Text']},
            remove_columns=raw_dataset['train'].column_names,
        )
        # train_texts = [text["text"] for text in raw_dataset["train"]]
        # test_texts = [text["text"] for text in raw_dataset["test"]]
    elif dataset == "wikitext":
        raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        raw_dataset = raw_dataset.map(
            lambda x: {'text': x['text']},
            remove_columns=raw_dataset['train'].column_names,
        )
        # train_texts = [text for text in raw_dataset["train"]["text"] if text.strip()]
        # test_texts = [text for text in raw_dataset["test"]["text"] if text.strip()]
    elif dataset == "code":
        raw_dataset = load_dataset("codeparrot/codeparrot-clean-train", split="train[:1%]")
        raw_dataset = raw_dataset.map(
            lambda x: {'text': x['content']},
            remove_columns=raw_dataset['train'].column_names,
        )
    elif dataset == "owt":
        # Load the OpenWebText dataset from Hugging Face
        raw_dataset = load_dataset("Skylion007/openwebtext", trust_remote_code=True, split='train[:1%]')
        # Create a DatasetDict with train/validation splits (90/10)
        split_idx = int(len(raw_dataset) * 0.9)
        train_split = raw_dataset.select(range(split_idx))
        test_split = raw_dataset.select(range(split_idx, len(raw_dataset)))
        raw_dataset = DatasetDict({
            'train': train_split,
            'test': test_split
        })
        # No need to rename columns as 'text' is already the field name
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    ## Initialize the tokenizer

    if char:
        char_int, eos_token_id = generate_char_vocab()
        vocab_size = len(char_int)
        def tokenize(text):
            if type(text['text']) == str:
                return {'tokenized': [char_int[c] for c in text['text']]}
            elif type(text['text']) == list:
                return {'tokenized': [[char_int[c] for c in t] for t in text["text"]]}
            else:
                raise Exception("Unknown type")
    else:
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        vocab_size = tokenizer.vocab_size
        eos_token_id = tokenizer.eos_token_id
        tokenize = lambda text: {'tokenized': tokenizer(text['text'], truncation=True, max_length=1024)['input_ids']}

    ## Tokenize the dataset

    raw_dataset = raw_dataset.map(
        tokenize,
        # remove_columns=raw_dataset['train'].column_names,
        num_proc=os.cpu_count(),
        batched=True
    )

    ## Batch examples into one long tensor

    def concatenate_ids(examples):
        # Flatten all ids and add EOS tokens
        all_ids = np.concatenate([np.array(ids + [eos_token_id]) for ids in examples["tokenized"] if ids])
        return {'ids': all_ids}

    dataset = raw_dataset.map(
        concatenate_ids,
        batched=True,
        remove_columns=raw_dataset['train'].column_names,
        num_proc=os.cpu_count()
    )

    # Build train and validation tensors
    dataset.set_format(type='torch', columns=['ids'])

    train_data = dataset["train"]["ids"]
    val_data = dataset["test"]["ids"]

    print(f"Train data size: {train_data.shape}, Val data size: {val_data.shape}")

    # If rank is provided, only keep the relevant shard of the data
    ## TODO: Do this first to save time.
    if rank is not None and world_size is not None:
        # Calculate shard size and indices for the training data
        train_size = len(train_data)
        shard_size = train_size // world_size
        start_idx = rank * shard_size
        end_idx = start_idx + shard_size if rank < world_size - 1 else train_size
        
        # Only keep the relevant shard for training data
        train_data = train_data[start_idx:end_idx]
        
        # Do not shard the validation set; keep the full validation data.

    # Cache the processed dataset shard
    cache_data = {
        "train": train_data,
        "val": val_data,
        "vocab_size": vocab_size,
    }
    # torch.save(cache_data, data_cache_file)

    return train_data, val_data, vocab_size

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--dataset", type=str, default="shakespeare",
                        help="Dataset: shakespeare, wikitext, or code")
    parser.add_argument("--char", action="store_true",
                        help="Enable character-level tokenization")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    args = parser.parse_args()

    train_data, val_data, vocab_size = get_dataset(args.dataset, 
                                                   args.block_size, 
                                                   char=args.char,
                                                   rank=args.rank,
                                                   world_size=args.world_size)
    print(train_data.shape, val_data.shape, vocab_size)
