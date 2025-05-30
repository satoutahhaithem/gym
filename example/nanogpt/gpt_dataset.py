import torch
import numpy as np
import os
import json

class NonContiguousGPTTrainDataset(torch.utils.data.Dataset):
    """Dataset for pre-segmented training data (2D tensor).
    Each row is an independent sequence with no continuity between examples.
    Suitable for datasets already divided into fixed-length chunks.
    """
    def __init__(self, data, device):
        assert data.ndim == 2
        self.examples, self.block_size = data.shape

        self.device = device

        self.data = torch.from_numpy(data).to(device=device).long()

    def __len__(self):
        return self.examples

    def __getitem__(self, idx):
        x = self.data[idx]
        return x[:-1], x[1:]

class LazyNonContiguousGPTTrainDataset(torch.utils.data.Dataset):
    """Dataset for pre-segmented training data with lazy loading.
    Chunks are cached locally but only loaded into memory when needed.
    Each row is an independent sequence with no continuity between examples.
    """
    def __init__(self, chunk_ids, cache_location, device, max_chunks_in_memory=None):
        self.chunk_ids = chunk_ids
        self.cache_location = cache_location
        self.device = device
        self.max_chunks_in_memory = max_chunks_in_memory
        
        # Try to read metadata for more efficient initialization
        metadata_file = os.path.join(cache_location, "cache_metadata.json")
        metadata = None
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                print(f"Using metadata from {metadata_file}")
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not read metadata file {metadata_file}: {e}")
                metadata = None
        
        # Build index mapping from global index to (chunk_id, local_idx)
        self.chunk_sizes = []
        self.chunk_offsets = []
        total_examples = 0
        
        # If we have metadata and all chunks are consecutive integers, use it for efficiency
        if (metadata and 
            chunk_ids == list(range(len(chunk_ids))) and 
            len(chunk_ids) <= metadata.get('num_chunks', 0)):
            
            print("Using metadata for efficient initialization")
            blocks_per_chunk = metadata.get('blocks_per_chunk', None)
            total_blocks = metadata.get('total_blocks', None)
            
            if blocks_per_chunk and total_blocks:
                # Use metadata to calculate chunk sizes without loading
                for i, chunk_id in enumerate(chunk_ids):
                    # Verify chunk file exists
                    cache_file = f'{cache_location}/chunk_{chunk_id}.npy'
                    if not os.path.exists(cache_file):
                        raise FileNotFoundError(f"Cached chunk file not found: {cache_file}")
                    
                    # Calculate expected chunk size from metadata
                    if i == len(chunk_ids) - 1 and i == metadata.get('num_chunks', 0) - 1:
                        # Last chunk of the full dataset - may have different size
                        chunk_size = total_blocks - (i * blocks_per_chunk)
                    else:
                        chunk_size = blocks_per_chunk
                    
                    self.chunk_sizes.append(chunk_size)
                    self.chunk_offsets.append(total_examples)
                    total_examples += chunk_size
                
                # Get block size from metadata or first chunk
                self.block_size = 1024  # Default from metadata, will verify below
                
                # Verify with first chunk to ensure consistency
                first_chunk_file = f'{cache_location}/chunk_{chunk_ids[0]}.npy'
                first_chunk_data = np.load(first_chunk_file)
                if first_chunk_data.ndim != 2:
                    raise ValueError(f"Expected 2D chunk data, got {first_chunk_data.ndim}D")
                self.block_size = first_chunk_data.shape[1]
                
                print(f"Initialized using metadata: {len(chunk_ids)} chunks, {total_examples} examples, block_size={self.block_size}")
            else:
                print("Metadata incomplete, falling back to chunk loading")
                metadata = None
        else:
            if metadata:
                print("Chunk IDs don't match metadata pattern, falling back to chunk loading")
            metadata = None
        
        # Fallback: load each chunk to get shape info (original behavior)
        if metadata is None:
            print("Loading chunks to determine sizes")
            for chunk_id in chunk_ids:
                cache_file = f'{cache_location}/chunk_{chunk_id}.npy'
                if not os.path.exists(cache_file):
                    raise FileNotFoundError(f"Cached chunk file not found: {cache_file}")
                
                # Load just to get shape, then discard
                chunk_data = np.load(cache_file)
                assert chunk_data.ndim == 2, f"Expected 2D chunk data, got {chunk_data.ndim}D"
                
                chunk_size = chunk_data.shape[0]
                self.chunk_sizes.append(chunk_size)
                self.chunk_offsets.append(total_examples)
                total_examples += chunk_size
                
                # Store block_size from first chunk
                if len(self.chunk_sizes) == 1:
                    self.block_size = chunk_data.shape[1]
        
        self.total_examples = total_examples
        self._loaded_chunks = {}  # Cache for loaded chunks
        self._chunk_access_order = []  # Track access order for LRU eviction
        
        print(f"Dataset initialized: {len(chunk_ids)} chunks, {self.total_examples} total examples")

    def __len__(self):
        return self.total_examples

    def _get_chunk_and_local_idx(self, global_idx):
        """Convert global index to (chunk_id, local_idx)"""
        for i, (chunk_id, offset, size) in enumerate(zip(self.chunk_ids, self.chunk_offsets, self.chunk_sizes)):
            if global_idx < offset + size:
                local_idx = global_idx - offset
                return chunk_id, local_idx
        raise IndexError(f"Index {global_idx} out of range")

    def _evict_old_chunks(self):
        """Remove old chunks from memory if we exceed the limit"""
        if self.max_chunks_in_memory is None:
            return
            
        while len(self._loaded_chunks) > self.max_chunks_in_memory:
            # Remove least recently used chunk
            oldest_chunk = self._chunk_access_order.pop(0)
            if oldest_chunk in self._loaded_chunks:
                del self._loaded_chunks[oldest_chunk]

    def _load_chunk(self, chunk_id):
        """Load chunk data if not already cached in memory"""
        if chunk_id not in self._loaded_chunks:
            # print(f'loading chunk {chunk_id}')
            cache_file = f'{self.cache_location}/chunk_{chunk_id}.npy'
            chunk_data = np.load(cache_file)
            self._loaded_chunks[chunk_id] = torch.from_numpy(chunk_data).to(device=self.device).long()
            
            # Evict old chunks if necessary
            self._evict_old_chunks()
        
        # Update access order for LRU
        if chunk_id in self._chunk_access_order:
            self._chunk_access_order.remove(chunk_id)
        self._chunk_access_order.append(chunk_id)
        
        return self._loaded_chunks[chunk_id]

    def __getitem__(self, idx):
        chunk_id, local_idx = self._get_chunk_and_local_idx(idx)
        chunk_data = self._load_chunk(chunk_id)
        x = chunk_data[local_idx]
        return x[:-1], x[1:]

    def get_memory_info(self):
        """Return information about current memory usage"""
        return {
            'chunks_in_memory': len(self._loaded_chunks),
            'max_chunks_in_memory': self.max_chunks_in_memory,
            'total_chunks': len(self.chunk_ids),
            'loaded_chunk_ids': list(self._loaded_chunks.keys())
        }

    @classmethod
    def from_cache(cls, cache_location, device, max_chunks_in_memory=None, chunk_range=None):
        """
        Create a LazyNonContiguousGPTTrainDataset from cached data using metadata.
        
        Args:
            cache_location: Directory containing cached chunks and metadata
            device: Device to load tensors on
            max_chunks_in_memory: Maximum chunks to keep in memory
            chunk_range: Tuple (start_chunk, end_chunk) to use subset of chunks, or None for all
            
        Returns:
            LazyNonContiguousGPTTrainDataset instance
        """
        metadata_file = os.path.join(cache_location, "cache_metadata.json")
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        num_chunks = metadata.get('num_chunks', 0)
        if num_chunks == 0:
            raise ValueError("No chunks found in metadata")
        
        if chunk_range is None:
            chunk_ids = list(range(num_chunks))
        else:
            start_chunk, end_chunk = chunk_range
            start_chunk = max(0, start_chunk)
            end_chunk = min(num_chunks, end_chunk)
            chunk_ids = list(range(start_chunk, end_chunk))
        
        print(f"Creating dataset from cache with {len(chunk_ids)} chunks (range: {chunk_ids[0] if chunk_ids else 'N/A'}-{chunk_ids[-1] if chunk_ids else 'N/A'})")
        
        return cls(chunk_ids, cache_location, device, max_chunks_in_memory)

    def get_cache_metadata(self):
        """Return metadata about the cached dataset"""
        metadata_file = os.path.join(self.cache_location, "cache_metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                return json.load(f)
        return None

class ContiguousGPTTrainDataset(torch.utils.data.Dataset):
    """Dataset for continuous token streams (1D tensor).
    Creates examples by sliding a window over the data.
    Preserves context and long-range dependencies in text.
    """
    def __init__(self, data, block_size, device):
        assert data.ndim == 1

        self.device = device

        self.data = torch.from_numpy(data).to(device=device).long()
        self.block_size = block_size

    def __len__(self):
        return self.data.shape[0] - self.block_size - 1

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.block_size + 1]
        return x[:-1], x[1:]