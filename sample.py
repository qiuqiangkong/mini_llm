"""
Modified from https://github.com/karpathy/nanoGPT/blob/master/sample.py
"""
from __future__ import annotations
import argparse
from pathlib import Path

import torch

from data.tokenizers import TokenizerChar
from train import TokenizerChar, get_model


def sample(args):

    # Arguments
    model_name = args.model_name
    ckpt_path = args.ckpt_path

    num_samples = 5  # Number of samples to draw
    max_new_ids = 256  # Number of IDs generated in each sample
    temperature = 1.0  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k = 200  # Retain only the top_k most likely IDs, clamp others to have 0 probability
    device = "cuda"

    # Paths
    root = "./assets/shakespeare_char"
    meta_path = Path(root, "meta.pkl")

    # Tokenizer
    tokenizer = TokenizerChar(meta_path=meta_path)

    # Load model
    model = get_model(model_name=model_name, vocab_size=len(tokenizer))
    model.load_state_dict(torch.load(ckpt_path))
    model.to(device)

    # Begin ID
    input_id = tokenizer.stoi("\n")  # 0
    input_ids = torch.LongTensor([[input_id]]).to(device)  # (b, 1)

    # Sample    
    for n in range(num_samples):

        with torch.no_grad():
            model.eval()
            ids = model.generate(
                ids=input_ids, 
                max_new_ids=max_new_ids, 
                temperature=temperature, 
                top_k=top_k
            )
            # shape: (b, t)

        ids = ids[0].cpu().numpy()
        strings = ids_to_text(ids, tokenizer)
        print(strings)
        print("------------")


def ids_to_text(ids, tokenizer):
    return "".join([tokenizer.itos(id) for id in ids])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    args = parser.parse_args()

    sample(args)