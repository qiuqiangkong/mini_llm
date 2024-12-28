"""
Modified from https://github.com/Lightning-AI/lit-llama/blob/main/lit_llama/model.py
"""
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from models.rope import build_rope, apply_rope


@dataclass
class LlamaConfig:
    block_size: int = 2048
    vocab_size: int = 32000  # Better to be divied by 64
    n_layer: int = 32
    n_head: int = 32
    n_embd: int = 4096


# Default Llama configurations
llama_configs = {
    "7B": dict(n_layer=32, n_head=32, n_embd=4096),
    "13B": dict(n_layer=40, n_head=40, n_embd=5120),
    "30B": dict(n_layer=60, n_head=52, n_embd=6656),
    "65B": dict(n_layer=80, n_head=64, n_embd=8192),
}


class Llama(nn.Module):
    r"""Llama model. Modified from https://github.com/Lightning-AI/lit-llama/blob/main/lit_llama/model.py"""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()

        self.config = config

        # word text embedding (wte)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        # Transformer blocks
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))

        # Output layers
        self.ln_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Build RoPE cache
        rope = build_rope(
            seq_len=self.config.block_size,
            head_dim=self.config.n_embd // self.config.n_head,
        )  # rope: (t, head_dim/2, 2)
        self.register_buffer(name="rope", tensor=rope)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer))
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer))

    def forward(
        self, 
        tokens: torch.Tensor,
        mask: None | torch.Tensor = None,
    ) -> torch.Tensor:
        r"""Next token prediction with GPT2.

        b: batch_size
        t: time_steps
        d: hidden_size
        v: vocab_size

        Args:
            tokens: (b, t)
            mask: None | (1, 1, t, t)

        Outputs:
            logits: (b, t, v)
        """
        
        device = tokens.device
        B, T = tokens.shape

        if mask is None:
            mask = build_causal_mask(seq_len=T).to(device)

        # Token embedding
        x = self.wte(tokens)  # shape: (b, t, d)

        # Transformer
        for block in self.blocks:
            x = block(x, self.rope, mask)
        # x: (b, t, d)

        # Output layers
        x = self.ln_f(x)  # shape: (b, t, d)
        logits = self.lm_head(x)  # shape: (b, t, v)

        return logits

    @torch.no_grad()
    def generate(
        self, 
        tokens: torch.LongTensor, 
        max_new_tokens: int, 
        temperature: float = 1.0, 
        top_k: None | int = None
    ):
        r"""
        Next token sampling with auto-regression. Make sure to use model.eval()

        b: batch_size
        t: time_steps
        v: vocab_size

        Args:
            tokens: (b, 1)
            max_new_tokens: int
            temperature: float
            top_k: None | int

        Returns:
            tokens: (b, t)
        """
        for _ in range(max_new_tokens):

            # If the sequence context is growing too long we must crop it at block_size
            if tokens.shape[1] <= self.config.block_size:
                prev_tokens = tokens
            else:
                prev_tokens = tokens[:, -self.config.block_size:]

            # Forward
            logits = self(prev_tokens)  # shape: (b, t, v)

            # Take the final step logits
            logits = logits[:, -1, :] / temperature  # shape: (b, v)

            # Crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)  # shape: (b, v)

            # Sample the next token
            next_token = torch.multinomial(probs, num_samples=1)  # shape: (b, 1)

            # Append the sampled token to the running tokens and continue
            tokens = torch.cat((tokens, next_token), dim=1)  # shape: (b, t)

        return tokens


class Block(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.att_norm = RMSNorm(config.n_embd)
        self.att = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        rope: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        r"""

        Args:
            x: (b, t, d)
            rope: (t, head_dim/2)
            mask: (1, 1, t, t)

        Outputs:
            x: (b, t, d)
        """
        x = x + self.att(self.att_norm(x), rope, mask)
        x = x + self.mlp(self.ffn_norm(x))
        return x


class RMSNorm(nn.Module):
    r"""Root Mean Square Layer Normalization.

    Ref: https://github.com/meta-llama/llama/blob/main/llama/model.py
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""RMSNorm.

        Args:
            x: (b, t, d)
           
        Outputs:
            x: (b, t, d)
        """
        norm_x = torch.mean(x ** 2, dim=-1, keepdim=True)
        output = x * torch.rsqrt(norm_x + self.eps) * self.scale
        return output


class CausalSelfAttention(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.block_size = config.block_size

    def forward(
        self,
        x: torch.Tensor,
        rope: torch.Tensor,
        mask: torch.Tensor,
        # max_seq_length: int,
        # input_pos: Optional[torch.Tensor] = None,
        # kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        r"""Causal self attention.

        b: batch size
        t: time steps
        d: latent dim
        h: heads num

        Args:
            x: (b, t, d)
            rope: (t, head_dim/2, 2)
            mask: (1, 1, )

        Outputs:
            x: (b, t, d)
        """

        B, T, D = x.shape

        # Calculate query, key, values
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # q, k, v shapes: (b, t, d)

        k = k.view(B, T, self.n_head, D // self.n_head)
        q = q.view(B, T, self.n_head, D // self.n_head)
        v = v.view(B, T, self.n_head, D // self.n_head)
        # q, k, v shapes: (b, t, h, d/h)

        q = apply_rope(q, rope)
        k = apply_rope(k, rope)
        # q, k shapes: (b, t, h, d/h)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # q, k, v shapes: (b, h, t, d/h)

        # Efficient attention using Flash Attention CUDA kernels
        x = F.scaled_dot_product_attention(
            query=q, 
            key=k, 
            value=v, 
            attn_mask=mask, 
            dropout_p=0.0
        )
        # shape: (b, h, t, d/h)

        x = x.transpose(1, 2).contiguous().view(B, T, D)  # shape: (b, t, d)

        # output projection
        x = self.c_proj(x)  # shape: (b, t, d)
        
        return x


class MLP(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()

        # The hyper-parameters follow https://github.com/Lightning-AI/lit-llama/blob/main/lit_llama/model.py
        hidden_dim = 4 * config.n_embd
        n_hidden = int(2 * hidden_dim / 3) 

        self.c_fc1 = nn.Linear(config.n_embd, n_hidden, bias=False)
        self.c_fc2 = nn.Linear(config.n_embd, n_hidden, bias=False)
        self.c_proj = nn.Linear(n_hidden, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Causal self attention.

        Args:
            x: (b, t, d)
           
        Outputs:
            x: (b, t, d)
        """
        x = F.silu(self.c_fc1(x)) * self.c_fc2(x)
        x = self.c_proj(x)
        return x


def build_causal_mask(seq_len: int) -> torch.Tensor:
    r"""Build causal mask."""
    ones = torch.ones((seq_len, seq_len), dtype=torch.bool)  # shape: (t, t)
    mask = torch.tril(ones)[None, None, :, :]  # shape: (1, 1, t, t)
    return mask