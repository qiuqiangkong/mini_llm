from __future__ import annotations

import random

import numpy as np
from torch.utils.data import Dataset
from typing_extensions import Literal


class ShakespeareChar(Dataset):
    r"""Shakespear dataset with plain texts. Size: 1 MB."""
    
    def __init__(self,
        text_path: str = "input.txt", 
        tokenizer: object = None,
        split: Literal["train", "val"] = "train",
        seq_len: int = 256,
    ):
        super().__init__()

        self.seq_len = seq_len

        # Load all texts
        self.tokens = load_text_to_tokens(text_path=text_path, tokenizer=tokenizer, split=split)

    def __getitem__(self, index: int) -> dict:
        r"""Fetch an input and a target for training. The `index` argument is 
        not used because we use only one book for training."""

        # Sample an audio name from all audios.
        idx = random.randint(0, len(self.tokens) - self.seq_len - 1)
        
        data = {
            "input": self.tokens[idx : idx + self.seq_len],
            "target": self.tokens[idx + 1 : idx + self.seq_len + 1],
        }

        return data

    def __len__(self):

        # We call 1000 steps as an `epoch`
        return 1000


def load_text_to_tokens(
    text_path: str, 
    tokenizer: object, 
    split: Literal["train", "val"]
) -> np.ndarray:
    r"""Load a text file and convert characters to tokens."""

    # Load texts
    with open(text_path, 'r') as file:
        text = file.read()

    # Convert texts to tokens
    tokens = np.array([tokenizer.stoi(char) for char in text])

    if split == "train":
        tokens = tokens[0 : 1003854]  # Consistent with nanoGPT

    elif split == "val":
        tokens = tokens[1003854 :]  # Consistent with nanoGPT

    else:
        raise ValueError(split)

    return tokens