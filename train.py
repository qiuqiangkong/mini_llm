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

from data.shakespeare import ShakespeareChar, load_text_to_ids
from tokenizers import Tokenizer


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
    model = get_model(model_name=model_name, vocab_size=len(tokenizer))
    model.to(device)

    # Optimizer
    optimizer = optim.AdamW(params=model.parameters(), lr=learning_rate)

    if wandb_log:
        wandb.init(project="mini_llm", name="{}".format(model_name))

    # Train
    for step, data in enumerate(tqdm(train_dataloader)):

        # Move data to device
        input_ids = data["id"][0 : -1].to(device)  # (b, t)
        target_ids = data["id"][1 :].to(device)  # (b, t)

        # Forward
        model.train()
        logits = model(ids=input_ids)  # shape: (b, t, vocab_size)

        # Loss
        loss = ce_loss(output=logits, target=target_ids)
        
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


class MySampler:
    def __init__(self, books_num: int):
        self.books_num = books_num
        
    def __iter__(self) -> int:
        while True:
            yield random.randint(a=0, b=self.books_num)


def get_model(model_name: str, vocab_size: int) -> nn.Module:

    if model_name == "GPT2":
        from models.gpt2 import GPTConfig, GPT2
        config = GPTConfig(
            block_size=1024,
            vocab_size=vocab_size,
            n_layer=12,
            n_head=12,
            n_embd=768
        )
        return GPT2(config=config)

    elif model_name == "Llama":
        from models.llama import LlamaConfig, Llama
        config = LlamaConfig(
            block_size=1024,
            vocab_size=vocab_size,
            n_layer=12,
            n_head=12,
            n_embd=768
        )
        return Llama(config=config)

    else:
        raise ValueError(model_name)


def ce_loss(output: torch.Tensor, target: torch.LongTensor) -> float:
    r"""Cross entropy loss.

    Args:
        output: (b, t, vocab_size)
        target: (b, t)

    Outputs:
        loss: torch.float
    """

    B, T, V = output.shape

    loss = F.cross_entropy(
        input=output.flatten(0, 1),  # shape: (b*t, vocab_size)
        target=target.flatten(0, 1),  # shape: (b*t,)
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
    ids = load_text_to_ids(text_path=text_path, tokenizer=tokenizer, split=split)

    losses = []

    for i in range(valid_steps):

        # Fetch data
        bgn = i * seq_len
        end = (i + 1) * seq_len + 1
        clip_ids = ids[bgn : end]  # shape: (t + 1,)

        input_ids = torch.LongTensor(clip_ids[None, 0 : -1]).to(device)  # (b, t)
        target_ids = torch.LongTensor(clip_ids[None, 1 :]).to(device)  # (b, t)

        # Forward
        with torch.no_grad():
            model.eval()
            logits = model(ids=input_ids)  # shape: (b, t, vocab_size)

        # Calculate loss
        loss = ce_loss(output=logits, target=target_ids)
        losses.append(loss.item())

    return np.mean(losses)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    args = parser.parse_args()

    train(args)