import torch
import argparse
import numpy as np
import math
import os
from datasets import load_dataset, Dataset, DatasetDict, load_dataset_builder

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

def get_dataset(dataset, block_size=1024, char=False, rank=None, world_size=None,
                dataset_proportion=None, val_ratio=0.1):
    """
    Loads and preprocesses the dataset with caching, using either a custom character-level tokenizer
    or the GPT2 tokenizer. It uses only a fraction of the full dataset, as controlled by dataset_proportion,
    then splits that fraction into training and validation portions (with val_ratio held out as validation).
    When rank and world_size are provided, the training portion is sharded among nodes.

    Args:
        dataset: a string identifier ("shakespeare", "wikitext", "code", or "owt")
        block_size: the sequence block size.
        char (bool): If True, use character-level tokenization; otherwise, use GPT-2 tokenization.
        rank (int, optional): The rank of the current process.
        world_size (int, optional): Total number of processes.
        dataset_proportion (float): Proportion of the dataset to use (0 to 1).
        val_ratio (float): Fraction of the used dataset to reserve for validation.
    """
    # Decide cache locations based on tokenization mode and rank.
    if char:
        cache_dir = os.path.join("cache", f"{dataset}_char")
    else:
        cache_dir = os.path.join("cache", dataset)
    os.makedirs(cache_dir, exist_ok=True)

    assert rank is not None
    rank_suffix = f"_rank{rank}_of{world_size}" if rank is not None else ""
    data_cache_file = os.path.join(cache_dir, f"data_block{block_size}_{rank_suffix}.pt")

    if os.path.exists(data_cache_file):
        print(f"Loading cached dataset from {data_cache_file}")
        cached_data = torch.load(data_cache_file)
        return cached_data["train"], cached_data["val"], cached_data["vocab_size"]

    print(f"Loading dataset: {dataset} {'(char-level)' if char else '(GPT2 tokenization)'}")
    
    # Determine the dataset identifier and mapping function.
    if dataset == "shakespeare":
        dataset_id = "Trelis/tiny-shakespeare"
        mapping_fn = lambda x: {'text': x['Text']}
        load_config = {}
        if dataset_proportion is None:
            dataset_proportion = 1.0
    elif dataset == "wikitext":
        dataset_id = "wikitext"
        config = "wikitext-2-raw-v1"
        mapping_fn = lambda x: {'text': x['text']}
        load_config = {"name": config}
        if dataset_proportion is None:
            dataset_proportion = 1.0
    elif dataset == "code":
        dataset_id = "codeparrot/codeparrot-clean-train"
        mapping_fn = lambda x: {'text': x['content']}
        load_config = {}
        if dataset_proportion is None:
            dataset_proportion = 0.1
    elif dataset == "owt":
        dataset_id = "Skylion007/openwebtext"
        mapping_fn = lambda x: x  # Assume openwebtext already has a 'text' field.
        load_config = {"trust_remote_code": True}
        if dataset_proportion is None:
            dataset_proportion = 0.01
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Use the dataset builder to obtain the total number of records.
    builder = load_dataset_builder(dataset_id, **load_config)
    total_records = builder.info.splits["train"].num_examples

    print(f"Total records in the train split: {total_records}")
    
    # Calculate the number of records to use and how to split them.
    used_records = int(total_records * dataset_proportion)
    train_records = int(used_records * (1 - val_ratio))
    val_records = used_records - train_records
    print(f"Using {used_records} records: {train_records} for training and {val_records} for validation")

    # Load only the desired subset by slicing with record numbers.
    if dataset == "wikitext":
        raw_train = load_dataset(dataset_id, config, split=f"train[:{used_records}]")
    else:
        raw_train = load_dataset(dataset_id, split=f"train[:{used_records}]", **load_config)

    # Apply mapping function if needed.
    if dataset != "owt":  # Assuming openwebtext already uses a 'text' field.
        raw_train = raw_train.map(mapping_fn, remove_columns=raw_train.column_names)

    # Split the dataset into training and validation portions.
    train_full = raw_train.select(range(train_records))
    val_full = raw_train.select(range(train_records, used_records))

    # For distributed training, shard the training set among nodes.
    if world_size is not None and world_size > 1:
        train_full = train_full.shard(num_shards=world_size, index=rank)

    raw_dataset = DatasetDict({"train": train_full, "test": val_full})

    ## Initialize the tokenizer.
    if char:
        char_int, eos_token_id = generate_char_vocab()
        vocab_size = len(char_int)
        def tokenize(example):
            text = example['text']
            if isinstance(text, str):
                return {'tokenized': [char_int[c] for c in text]}
            elif isinstance(text, list):
                return {'tokenized': [[char_int[c] for c in t] for t in text]}
            else:
                raise Exception("Unknown type")
    else:
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        vocab_size = tokenizer.vocab_size
        eos_token_id = tokenizer.eos_token_id
        def tokenize(example):
            return {'tokenized': tokenizer(example['text'], truncation=True, max_length=block_size)['input_ids']}

    ## Tokenize the dataset.
    raw_dataset = raw_dataset.map(
        tokenize,
        num_proc=os.cpu_count(),
        batched=True
    )

    ## Concatenate tokens and append EOS tokens.
    def concatenate_ids(examples):
        all_ids = np.concatenate([np.array(ids + [eos_token_id]) for ids in examples["tokenized"] if ids])
        return {'ids': all_ids}

    dataset_processed = raw_dataset.map(
        concatenate_ids,
        batched=True,
        remove_columns=raw_dataset['train'].column_names,
        num_proc=os.cpu_count()
    )

    dataset_processed.set_format(type='torch', columns=['ids'])

    train_data = dataset_processed["train"]["ids"]
    val_data = dataset_processed["test"]["ids"]

    print(f"Train data size: {train_data.shape}, Val data size: {val_data.shape}")

    cache_data = {
        "train": train_data,
        "val": val_data,
        "vocab_size": vocab_size,
    }
    # Optionally cache the processed data:
    torch.save(cache_data, data_cache_file)

    return train_data, val_data, vocab_size

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--dataset", type=str, default="shakespeare",
                        help="Dataset: shakespeare, wikitext, code, or owt")
    parser.add_argument("--char", action="store_true",
                        help="Enable character-level tokenization")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--dataset_proportion", type=float, default=1.0,
                        help="Proportion of the dataset to use (0 to 1)")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Fraction of the used dataset to reserve for validation")
    args = parser.parse_args()

    if args.rank is not None:
        # Single rank execution
        train_data, val_data, vocab_size = get_dataset(args.dataset, 
                                                       args.block_size, 
                                                       char=args.char,
                                                       rank=args.rank,
                                                       world_size=args.world_size,
                                                       dataset_proportion=args.dataset_proportion,
                                                       val_ratio=args.val_ratio)
        print(f"Rank {args.rank}:", train_data.shape, val_data.shape, vocab_size)
    else:
        # Run for all ranks sequentially
        for rank in range(args.world_size):
            train_data, val_data, vocab_size = get_dataset(args.dataset, 
                                                           args.block_size, 
                                                           char=args.char,
                                                           rank=rank,
                                                           world_size=args.world_size,
                                                           dataset_proportion=args.dataset_proportion,
                                                           val_ratio=args.val_ratio)
            print(f"Rank {rank}:", train_data.shape, val_data.shape, vocab_size)
