import re
from typing import Dict, List

import tiktoken

class SimpleTokenizerV1:
    def __init__(self, vocab_file: Dict[str, int]):
        self.str_to_int = vocab_file
        self.int_to_str = {v: k for k, v in vocab_file.items()}
    
    def encode(self, text: str) -> List[int]:
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)

        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item.lower() if item in self.str_to_int else '<|unk|>' for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids: List[int]) -> str:
        text = " ".join([self.int_to_str[i] for i in ids])

        # Remove spaces before these characters
        text = re.sub(r'\s+([,.?_!])', r'\1', text)
        return text

# BPE breaks down words not within vocabulary into subword tokens
tokenizer = tiktoken.get_encoding("gpt2")
