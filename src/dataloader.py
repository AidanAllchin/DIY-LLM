from typing import List, Tuple
from pathlib import Path
import json

import torch
from torch.utils.data import Dataset, DataLoader as TorchDL, IterableDataset
from src.tokenization import tokenizer
from collections import deque

project_root = Path(__file__).parent.parent

# Read the text files, concatenate them, separate with endoftext tokens, and 
# then return as many tokens as are requested

def load_raw_tokens(num_tokens: int) -> List[int]:
    """Load and tokenize JSON objects until num_tokens is reached."""
    data_dir = project_root / "data"
    # Collect JSON files in sorted order
    files = sorted(data_dir.glob("*.json"))
    tokens: List[int] = []
    for file in files:
        print(f"Loading {file}")
        try:
            with open(file, 'r', encoding='utf-8') as f:
                objects = json.load(f)
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue
        if not isinstance(objects, list):
            print(f"Skipping {file}: expected JSON array")
            continue
        for obj in objects:
            if not isinstance(obj, dict) or "text" not in obj:
                continue
            text = obj["text"] + "<|endoftext|>"

            encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
            if encoded:
                tokens.extend(encoded)
            if len(tokens) >= num_tokens:
                return tokens[:num_tokens]
        #print(f"Accumulated {len(tokens)} tokens so far")
    return tokens[:num_tokens]

class StaticDataset(Dataset):
    def __init__(self, tokenized: List[int], max_length: int, stride: int):
        # self.tokenized = tokenized
        self.max_length = max_length
        self.stride = stride
        
        self.input_ids  = []
        self.target_ids = []

        for i in range(0, len(tokenized) - max_length, stride):
            input_chunk  = tokenized[i:i + max_length]
            target_chunk = tokenized[i + 1:i + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self) -> int:
        return len(self.input_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Returns a single batch of input and target tokens """
        return self.input_ids[idx], self.target_ids[idx]

class StreamingDataset(IterableDataset):
    """Iterable dataset that streams JSON files and yields (input, target) windows."""
    def __init__(self, data_dir: Path, max_length: int, stride: int):
        self.files = sorted(data_dir.glob("*.json"))
        self.max_length = max_length
        self.stride = stride

    def __iter__(self):
        buffer = deque()
        for file in self.files:
            try:
                objects = json.load(open(file, 'r', encoding='utf-8'))
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue
            if not isinstance(objects, list):
                continue
            for obj in objects:
                if not isinstance(obj, dict) or "text" not in obj:
                    continue
                text = obj["text"] + "<|endoftext|>"
                toks = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
                for tok in toks:
                    buffer.append(tok)
                    if len(buffer) >= self.max_length + 1:
                        window = list(buffer)
                        input_ids  = torch.tensor(window[:self.max_length])
                        target_ids = torch.tensor(window[1:self.max_length+1])

                        yield input_ids, target_ids
                        
                        # Slide the window forward by stride
                        for _ in range(self.stride):
                            if buffer:
                                buffer.popleft()

def create_dataloader(
        batch_size: int = 4,
        max_length: int = 256,
        stride: int = 128,
        shuffle: bool = False,
        drop_last: bool = True,
        num_workers: int = 0
    ) -> TorchDL:
    """
    Create a DataLoader instance

    Args:
        batch_size (int): Batch size
        max_length (int): Maximum sequence length
        stride (int): Stride between sequences
        shuffle (bool): Whether to shuffle the dataset
        drop_last (bool): Whether to drop the last batch if it is not full
        num_workers (int): Number of workers to use for data loading

    Returns:
        DataLoaderV1: A DataLoaderV1 instance
    """
    print(f"Streaming dataset with max_length={max_length}, stride={stride}")
    dataset = StreamingDataset(project_root / "data", max_length, stride)
    # Wrap in PyTorch DataLoader
    return TorchDL(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
