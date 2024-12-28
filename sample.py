"""
Modified from https://github.com/karpathy/nanoGPT/blob/master/sample.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from train import Tokenizer, get_model


def sample(args):

    # Arguments
    model_name = args.model_name
    ckpt_path = args.ckpt_path

    start_char = "\n"
    num_samples = 5  # Number of samples to draw
    max_new_tokens = 256  # Number of tokens generated in each sample
    temperature = 1.0  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k = 200  # Retain only the top_k most likely tokens, clamp others to have 0 probability
    device = "cuda"

    # Paths
    root = "./datasets/shakespeare_char"
    meta_path = Path(root, "meta.pkl")

    # Load model
    model = get_model(model_name)
    model.load_state_dict(torch.load(ckpt_path))
    model.to(device)

    tokenizer = Tokenizer(meta_path=meta_path)

    token = tokenizer.stoi(start_char)
    input_tokens = torch.LongTensor([[token]]).to(device)  # (b, 1)

    # Sample    
    for n in range(num_samples):

        with torch.no_grad():
            model.eval()
            tokens = model.generate(
                tokens=input_tokens, 
                max_new_tokens=max_new_tokens, 
                temperature=temperature, 
                top_k=top_k
            )
            # shape: (b, t)

        tokens = tokens[0].cpu().numpy()
        strings = tokens_to_string(tokens, tokenizer)
        print(strings)
        print('---------------')


def tokens_to_string(tokens, tokenizer):
    return "".join([tokenizer.itos(token) for token in tokens])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    args = parser.parse_args()

    sample(args)