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
from accelerate import Accelerator
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing_extensions import Literal

from data.shakespeare import ShakespeareChar, load_text_to_ids
from data.tokenizers import TokenizerChar
from train import MySampler, get_model, ce_loss, validate


def train(args):

    # Arguments
    model_name = args.model_name
    wandb_log = not args.no_log

    # Default parameters
    batch_size = 16
    num_workers = 16
    pin_memory = True
    learning_rate = 1e-4
    test_every_n_steps = 200
    save_every_n_steps = 2000
    training_steps = 10000
    seq_len = 256
    device = "cuda"

    filename = Path(__file__).stem

    # Paths
    root = "./assets/shakespeare_char"
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
    optimizer = optim.AdamW(params=model.parameters(), lr=learning_rate)

    # Prepare for multiprocessing
    accelerator = Accelerator()
    
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader)

    # Logger
    if wandb_log and accelerator.is_main_process:
        wandb.init(project="mini_llm", name="{}".format(model_name))

    # Train
    for step, data in enumerate(tqdm(train_dataloader)):

        # Move data to device
        input_ids = data["id"][:, 0 : -1]  # (b, t)
        target_ids = data["id"][:, 1 :]  # (b, t)

        # Forward
        model.train()
        logits = model(ids=input_ids)  # shape: (b, t, vocab_size)

        # Loss
        loss = ce_loss(output=logits, target=target_ids)
        
        # Optimize
        optimizer.zero_grad()  # Reset all parameter.grad to 0
        accelerator.backward(loss)  # Update all parameter.grad
        optimizer.step()  # Update all parameters based on all parameter.grad

        # Evaluate
        if step % test_every_n_steps == 0 and accelerator.is_main_process:
            
            loss_dict = {}

            for split in ["train", "val"]:
                loss = validate(
                    text_path=text_path,
                    tokenizer=tokenizer,
                    split=split,
                    model=accelerator.unwrap_model(model),
                    seq_len=seq_len
                )
                loss_dict[split] = loss

            print("Train loss: {}".format(loss_dict["train"]))
            print("Test loss: {}".format(loss_dict["val"]))

            if wandb_log:
                wandb.log(
                    data={"train_loss": loss_dict["train"], "test_loss": loss_dict["val"]},
                    step=step
                )

        # Save model
        if step % save_every_n_steps == 0 and accelerator.is_main_process:
            ckpt_path = Path(ckpts_dir, "step={}.pth".format(step))
            torch.save(accelerator.unwrap_model(model).state_dict(), ckpt_path)
            print("Save model to {}".format(ckpt_path))

        if step == training_steps:
            break
        


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--no_log', action='store_true', default=False)
    args = parser.parse_args()

    train(args)