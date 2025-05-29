import numpy as np
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from exogym.example.nanogpt.build_dataset import build_dataset
from exogym.example.nanogpt.gpt_dataset import ContiguousGPTTrainDataset, LazyNonContiguousGPTTrainDataset

def count_files_in_s3_folder(bucket_name, folder_prefix, s3_client):
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=folder_prefix)

    file_count = sum(1 for page in pages for _ in page.get('Contents', []))
    
    return file_count

def load_chunk(chunk_id, s3_client):
    cache_location = f'cache/s3/owt/'
    if not os.path.exists(cache_location):
        os.makedirs(cache_location, exist_ok=True)

    cache_file = f'{cache_location}/chunk_{chunk_id}.npy'
    if os.path.exists(cache_file):
        return np.load(cache_file)
    else:
        s3_client.download_file(Bucket='exo-datasets', Key=f'owt/chunk_{chunk_id}.npy', Filename=cache_file)
        return np.load(cache_file)

def load_data(start_pc, end_pc):
    import boto3
    s3_client = boto3.client('s3')

    chunk_count = count_files_in_s3_folder('exo-datasets', 'owt/', s3_client)

    chunk_ids = np.arange(chunk_count)
    chunk_ids = chunk_ids[int(start_pc * chunk_count):int(end_pc * chunk_count)]
    print(f' importing {len(chunk_ids)} chunks [{chunk_ids[0]},{chunk_ids[-1]}]')
    data = []
    for chunk_id in tqdm(chunk_ids):
        data.append(load_chunk(chunk_id, s3_client))
    return np.concatenate(data)

def load_data_concurrent(start_pc, end_pc, max_workers=8):
    import boto3
    s3_client = boto3.client('s3')
    chunk_count = count_files_in_s3_folder('exo-datasets', 'owt/', s3_client)
    chunk_ids = np.arange(chunk_count)
    chunk_ids = chunk_ids[int(start_pc * chunk_count):int(end_pc * chunk_count)]

    print(f'Importing {len(chunk_ids)} chunks in up to {max_workers} threads…')

    data = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        # submit all jobs
        futures = {pool.submit(load_chunk, int(cid), s3_client): cid for cid in chunk_ids}

        # as each one completes, append its result
        for f in tqdm(as_completed(futures), total=len(futures)):
            data.append(f.result())

    return np.concatenate(data)

def preload_chunks_to_cache(start_pc, end_pc, max_workers=8):
    """Preload chunks to local cache without loading them all into memory."""
    import boto3
    s3_client = boto3.client('s3')
    chunk_count = count_files_in_s3_folder('exo-datasets', 'owt/', s3_client)
    chunk_ids = np.arange(chunk_count)
    chunk_ids = chunk_ids[int(start_pc * chunk_count):int(end_pc * chunk_count)]

    print(f'Preloading {len(chunk_ids)} chunks to cache in up to {max_workers} threads…')

    cache_location = f'cache/s3/owt/'
    if not os.path.exists(cache_location):
        os.makedirs(cache_location, exist_ok=True)

    # Check which chunks need to be downloaded
    chunks_to_download = []
    for chunk_id in chunk_ids:
        cache_file = f'{cache_location}/chunk_{chunk_id}.npy'
        if not os.path.exists(cache_file):
            chunks_to_download.append(chunk_id)

    if chunks_to_download:
        print(f'Downloading {len(chunks_to_download)} missing chunks…')
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            # Submit download jobs for missing chunks
            futures = {pool.submit(load_chunk, int(cid), s3_client): cid for cid in chunks_to_download}

            # Wait for all downloads to complete
            for f in tqdm(as_completed(futures), total=len(futures)):
                f.result()  # This will download and cache the chunk
    else:
        print('All chunks already cached locally.')

    return chunk_ids, cache_location

def get_dataset(dataset_name, block_size, device, start_pc=0.0, end_pc=1.0, max_workers=8, max_chunks_in_memory=None):
    if dataset_name != 'owt':
        data, vocab_size = build_dataset(dataset_name, block_size, start_pc=start_pc, end_pc=end_pc)

        dataset = ContiguousGPTTrainDataset(data, block_size=block_size, device=device)
    else:
        try:
            import boto3
        except ImportError:
            raise ImportError("boto3 is not installed. Please install it using `pip install boto3`.")

        # For OWT, preload chunks to cache but use lazy loading
        chunk_ids, cache_location = preload_chunks_to_cache(start_pc, end_pc, max_workers=max_workers)
        vocab_size = 50257

        dataset = LazyNonContiguousGPTTrainDataset(chunk_ids, cache_location, device=device, max_chunks_in_memory=max_chunks_in_memory)

    return dataset, vocab_size