from __future__ import annotations

import argparse
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing_extensions import Literal

from data.shakespeare import ShakespeareChar, load_text_to_tokens


def train(args):

    # Arguments
    model_name = args.model_name

    # Default parameters
    batch_size = 16
    num_workers = 16
    pin_memory = True
    learning_rate = 1e-4
    test_every_n_steps = 200
    save_every_n_steps = 2000
    training_steps = 10000
    wandb_log = True
    device = "cuda"
    seq_len = 256

    filename = Path(__file__).stem

    # Paths
    root = "./datasets/shakespeare_char"
    text_path = Path(root, "input.txt")
    meta_path = Path(root, "meta.pkl")

    # Checkpoints directory
    ckpts_dir = Path("./checkpoints", filename, model_name)
    Path(ckpts_dir).mkdir(parents=True, exist_ok=True)

    # Tokenizer
    tokenizer = Tokenizer(meta_path=meta_path)

    # Dataset
    train_dataset = ShakespeareChar(
        text_path=text_path,
        tokenizer=tokenizer,
        split="train",
        seq_len=seq_len
    )

    # Sampler
    train_sampler = MySampler(books_num=1)

    # Dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        num_workers=num_workers, 
        pin_memory=pin_memory
    )

    # Model
    model = get_model(model_name)
    model.to(device)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    if wandb_log:
        wandb.init(project="mini_llm", name="{}".format(model_name))

    # Train
    for step, data in enumerate(tqdm(train_dataloader)):

        # Move data to device
        input_tokens = data["input"].to(device)  # (b, t)
        target_tokens = data["target"].to(device)  # (b, t)

        # Forward
        model.train()
        output = model(tokens=input_tokens)  # shape: (b, t, vocab_size)

        # Loss
        loss = bce_loss(output=output, target=target_tokens)
        
        # Optimize
        optimizer.zero_grad()   # Reset all parameter.grad to 0
        loss.backward()     # Update all parameter.grad
        optimizer.step()    # Update all parameters based on all parameter.grad

        if step % test_every_n_steps == 0:
            
            train_loss = validate(
                text_path=text_path,
                tokenizer=tokenizer,
                split="train",
                model=model, 
                seq_len=seq_len
            )
            test_loss = validate(
                text_path=text_path,
                tokenizer=tokenizer,
                split="val",
                model=model, 
                seq_len=seq_len
            )
            print("------ step: {} ------".format(step))
            print("Train loss: {}".format(train_loss))
            print("Test loss: {}".format(test_loss))

            if wandb_log:
                wandb.log(
                    data={"train_loss": train_loss, "test_loss": test_loss},
                    step=step
                )

        # Save model
        if step % save_every_n_steps == 0:
            ckpt_path = Path(ckpts_dir, "step={}.pth".format(step))
            torch.save(model.state_dict(), ckpt_path)
            print("Save model to {}".format(ckpt_path))

            ckpt_path = Path(ckpts_dir, "latest.pth")
            torch.save(model.state_dict(), Path(ckpt_path))
            print("Save model to {}".format(ckpt_path))

        if step == training_steps:
            break


class Tokenizer:
    def __init__(self, meta_path: str):
        
        with open(meta_path, 'rb') as f:
            self.meta = pickle.load(f)

    def stoi(self, char: str) -> int:
        return self.meta["stoi"][char]

    def itos(self, index: int) -> str:
        return self.meta["itos"][index]


class MySampler:
    def __init__(self, books_num: int):
        self.books_num = books_num
        
    def __iter__(self) -> int:
        while True:
            yield random.randint(a=0, b=self.books_num)


def get_model(model_name: str) -> nn.Module:

    if model_name == "GPT2":
        from models.gpt2 import GPTConfig, GPT2
        config = GPTConfig(
            block_size=1024,
            vocab_size=50304,
            n_layer=12,
            n_head=12,
            n_embd=768
        )
        return GPT2(config=config)

    elif model_name == "Llama":
        from models.llama import LlamaConfig, Llama
        config = LlamaConfig(
            block_size=1024,
            vocab_size=50304,
            n_layer=12,
            n_head=12,
            n_embd=768
        )
        return Llama(config=config)

    else:
        raise NotImplementedError(model_name)


def bce_loss(output: torch.Tensor, target: torch.LongTensor) -> float:
    r"""

    Args:
        output: (b, t, vocab_size)
        target: (b, t)

    Outputs:
        loss: torch.float
    """

    B, T, V = output.shape

    loss = F.cross_entropy(
        input=output.view(B * T, V), 
        target=target.view(B * T), 
        ignore_index=-1
    )

    return loss


def validate(
    text_path: str,
    tokenizer: object,
    split: Literal["train", "val"],
    model: nn.Module,
    seq_len: int,
    valid_steps: int = 100
) -> float:
    r"""Validate the model on part of data."""

    device = next(model.parameters()).device

    # Load tokens
    tokens = load_text_to_tokens(text_path=text_path, tokenizer=tokenizer, split=split)

    losses = []

    for i in range(valid_steps):

        # Fetch data
        bgn = i * seq_len
        end = (i + 1) * seq_len

        input_tokens = torch.LongTensor(tokens[None, bgn : end]).to(device)  # (b, t)
        target_tokens = torch.LongTensor(tokens[None, bgn + 1 : end + 1]).to(device)  # (b, t)

        # Forward
        with torch.no_grad():
            model.eval()
            output = model(tokens=input_tokens)  # shape: (b, t, vocab_size)

        # Calculate loss
        loss = bce_loss(output=output, target=target_tokens)
        losses.append(loss.item())

    return np.mean(losses)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    args = parser.parse_args()

    train(args)