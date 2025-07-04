from __future__ import annotations

import random

import numpy as np
from torch.utils.data import Dataset
from typing import Literal


class ShakespeareChar(Dataset):
    r"""Shakespear dataset with plain texts. Size: 1 MB."""
    
    def __init__(self,
        text_path: str = "input.txt", 
        tokenizer: object = None,
        split: Literal["train", "test"] = "train",
        seq_len: int = 256,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.ids = load_text_to_ids(text_path=text_path, tokenizer=tokenizer, split=split)

    def __getitem__(self, index: int) -> dict:
        r"""The `index` argument is not used because we use only one book for training."""

        # Randomly sample a position in the book
        idx = random.randint(0, len(self.ids) - self.seq_len - 1)
        
        data = {
            "input_id": self.ids[idx : idx + self.seq_len],
            "target_id": self.ids[idx + 1 : idx + self.seq_len + 1]
        }

        return data

    def __len__(self):
        return 1000  # We call 1000 steps as an `epoch`


def load_text_to_ids(
    text_path: str, 
    tokenizer: object, 
    split: Literal["train", "test"]
) -> np.ndarray:
    r"""Load a text file and convert characters to tokens."""

    # Load texts
    with open(text_path, 'r') as file:
        text = file.read()

    # Convert texts to token IDs
    ids = np.array([tokenizer.stoi(char) for char in text])

    if split == "train":
        ids = ids[0 : 1003854]  # Consistent with nanoGPT

    elif split == "test":
        ids = ids[1003854 :]  # Consistent with nanoGPT

    else:
        raise ValueError(split)

    return ids