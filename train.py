from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import LongTensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing_extensions import Literal

import wandb
from mini_llm.datasets.shakespeare import ShakespeareChar, load_text_to_ids
from mini_llm.losses import ce_loss
from mini_llm.samplers import MySampler
from mini_llm.tokenizers.char import TokenizerChar


def train(args) -> None:

    # Arguments
    model_name = args.model_name
    wandb_log = not args.no_log

    # Default parameters
    batch_size = 16
    num_workers = 16
    pin_memory = True
    lr = 1e-4
    test_every_n_steps = 200
    save_every_n_steps = 2000
    training_steps = 10000
    seq_len = 256
    device = "cuda"

    filename = Path(__file__).stem

    # Paths
    root = "./datasets/shakespeare_char"
    text_path = Path(root, "input.txt")
    meta_path = Path(root, "meta.pkl")

    # Checkpoints directory
    ckpts_dir = Path("./checkpoints", filename, model_name)
    Path(ckpts_dir).mkdir(parents=True, exist_ok=True)

    # Tokenizer
    tokenizer = TokenizerChar(meta_path=meta_path)

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
    optimizer = optim.AdamW(params=model.parameters(), lr=lr)

    # Logger
    if wandb_log:
        wandb.init(project="mini_llm", name=model_name)

    # Train
    for step, data in enumerate(tqdm(train_dataloader)):

        # Move data to device
        input_ids = data["input_id"].to(device)  # (b, l)
        target_ids = data["target_id"].to(device)  # (b, l)
        
        # Forward
        model.train()
        logits = model(ids=input_ids)  # (b, l, vocab_size)

        # Loss
        loss = ce_loss(output=logits, target=target_ids)
        
        # Optimize
        optimizer.zero_grad()  # Reset all parameter.grad to 0
        loss.backward()  # Update all parameter.grad
        optimizer.step()  # Update all parameters based on all parameter.grad

        # Evaluate
        if step % test_every_n_steps == 0:

            loss_dict = {}

            for split in ["train", "test"]:
                loss = validate(
                    text_path=text_path,
                    tokenizer=tokenizer,
                    split=split,
                    model=model,
                    seq_len=seq_len
                )
                loss_dict[split] = loss

            if wandb_log:
                wandb.log(
                    data={"train_loss": loss_dict["train"], "test_loss": loss_dict["test"]},
                    step=step
                )

            print("Train loss: {}".format(loss_dict["train"]))
            print("Test loss: {}".format(loss_dict["test"]))

        # Save model
        if step % save_every_n_steps == 0:
            ckpt_path = Path(ckpts_dir, f"step={step}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Save model to {ckpt_path}")

        if step == training_steps:
            break


def get_model(model_name: str, vocab_size: int) -> nn.Module:

    if model_name == "GPT2":
        from mini_llm.models.gpt2 import GPT2, GPTConfig
        config = GPTConfig(
            block_size=1024,
            vocab_size=vocab_size,
            n_layer=12,
            n_head=12,
            n_embd=768
        )  # 82M params
        return GPT2(config=config)

    elif model_name == "Llama":
        from mini_llm.models.llama import Llama, LlamaConfig
        config = LlamaConfig(
            block_size=1024,
            vocab_size=vocab_size,
            n_layer=12,
            n_head=12,
            n_embd=768
        )  # 81M params
        return Llama(config=config)

    else:
        raise ValueError(model_name)


def validate(
    text_path: str,
    tokenizer: object,
    split: Literal["train", "test"],
    model: nn.Module,
    seq_len: int,
    valid_steps: int = 100
) -> float:
    r"""Validate the model on part of data."""

    device = next(model.parameters()).device

    # Load tokens
    ids = load_text_to_ids(text_path=text_path, tokenizer=tokenizer, split=split)

    losses = []

    # Sample indexes from the beginning
    for i in range(valid_steps):

        bgn = i * seq_len
        input_ids = ids[bgn : bgn + seq_len]
        target_ids = ids[bgn + 1 : bgn + seq_len + 1]

        input_ids = LongTensor(input_ids[None, :]).to(device)  # (b, l)
        target_ids = LongTensor(target_ids[None, :]).to(device)  # (b, l)

        # Forward
        with torch.no_grad():
            model.eval()
            logits = model(ids=input_ids)  # shape: (b, l, vocab_size)

        # Calculate loss
        loss = ce_loss(output=logits, target=target_ids)
        losses.append(loss.item())

    return np.mean(losses)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--no_log', action='store_true', default=False)
    args = parser.parse_args()

    train(args)