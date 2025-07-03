from dataclasses import dataclass

import torch
import torch.nn as nn
from einops import rearrange
from torch import LongTensor, Tensor
from torch.nn import functional as F

from mini_llm.models.rope import RoPE


@dataclass
class LlamaConfig:
    # 7B configuration
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
    r"""Llama model. Modified from:

    https://github.com/Lightning-AI/lit-llama/blob/main/lit_llama/model.py
    """
    
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()

        self.block_size = config.block_size

        # ID embedder
        self.id_embedder = nn.Embedding(config.vocab_size, config.n_embd)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            Block(config.n_embd, config.n_head) for _ in range(config.n_layer)
        )

        # Output layers
        self.final_norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # RoPE cache
        head_dim = config.n_embd // config.n_head
        self.rope = RoPE(head_dim, max_len=config.block_size)
    
    def forward(
        self, 
        ids: LongTensor,
        mask: None | Tensor = None,
    ) -> Tensor:
        r"""Next token prediction with Llama.

        b: batch_size
        l: seq_len
        d: hidden_size
        v: vocab_size

        Args:
            IDs: (b, l)
            mask: None | (1, 1, l, l)

        Outputs:
            logits: (b, l, v)
        """
        
        device = ids.device
        B, L = ids.shape

        assert L <= self.block_size, f"Can not forward sequence of {L} > {self.block_size}"

        if mask is None:
            mask = build_causal_mask(seq_len=L).to(device)

        # IDs embedding
        x = self.id_embedder(ids)  # (b, l, d)

        # Transformer
        for block in self.blocks:
            x = block(x, self.rope, mask=mask)  # (b, l, d)

        # Output layers
        x = self.final_norm(x)  # (b, l, d)
        logits = self.lm_head(x)  # (b, l, v)

        return logits

    @torch.no_grad()
    def generate(
        self, 
        ids: LongTensor, 
        max_new_ids: int, 
        temperature: float = 1.0, 
        top_k: None | int = None
    ) -> LongTensor:
        r"""Next ID sampling with auto-regression.

        b: batch_size
        l: seq_len
        v: vocab_size

        Args:
            ids: (b, 1)
            max_new_ids: int
            temperature: float
            top_k: None | int

        Returns:
            new_ids: (b, l), sampled IDs
        """
        input_len = ids.shape[1]

        for _ in range(max_new_ids):

            # If the sequence context is growing too long we must crop it at block_size
            if ids.shape[1] <= self.block_size:
                prev_ids = ids
            else:
                prev_ids = ids[:, -self.block_size:]

            # Forward
            logits = self(prev_ids)  # shape: (b, l, v)

            # Take the final step logits
            logits = logits[:, -1, :] / temperature  # shape: (b, v)

            # Crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)  # shape: (b, v)

            # Sample the next ID
            next_id = torch.multinomial(probs, num_samples=1)  # shape: (b, 1)

            # Append the sampled ID to the running IDs and continue
            ids = torch.cat((ids, next_id), dim=1)  # shape: (b, t)

        new_ids = ids[:, input_len:]  # shape: (b, t)

        return new_ids


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int) -> None:
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.norm2 = RMSNorm(n_embd)
        self.mlp = MLP(n_embd, n_head)

    def forward(
        self,
        x: Tensor,
        rope: Tensor, 
        mask: Tensor
    ) -> Tensor:
        r"""

        b: batch_size
        l: seq_len
        d: n_embd

        Args:
            x: (b, l, d)
            rope: (l, head_dim/2)
            mask: (1, 1, l, l)

        Outputs:
            x: (b, l, d)
        """

        x = x + self.attn(self.norm1(x), rope, mask)
        x = x + self.mlp(self.norm2(x))
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
    def __init__(self, n_embd: int, n_head: int) -> None:
        super().__init__()

        assert n_embd % n_head == 0
        self.head_dim = n_embd // n_head

        self.qkv_linear = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(
        self,
        x: Tensor,
        rope: Tensor,
        mask: Tensor,
    ) -> torch.Tensor:
        r"""Causal self attention.

        b: batch_size
        l: seq_len
        d: latent_dim
        n: n_head
        h: head_dim

        Args:
            x: (b, l, d)
            rope: (l, head_dim/2, 2)
            mask: (1, 1)

        Outputs:
            x: (b, l, d)
        """

        B, L, D = x.shape
        
        # Calculate query, key, values
        q, k, v = self.qkv_linear(x).chunk(chunks=3, dim=2)  # shapes: (b, l, d)
        q = rearrange(q, 'b l (n h) -> b l n h', h=self.head_dim)  # (b, l, n, h)
        k = rearrange(k, 'b l (n h) -> b l n h', h=self.head_dim)  # (b, l, n, h)
        v = rearrange(v, 'b l (n h) -> b l n h', h=self.head_dim)  # (b, l, n, h)

        # Apply RoPE
        q = rope(q)  # (b, l, n, h)
        k = rope(k)  # (b, l, n, h)

        # Efficient attention using Flash Attention CUDA kernels
        x = F.scaled_dot_product_attention(
            query=rearrange(q, 'b l n h -> b n l h'), 
            key=rearrange(k, 'b l n h -> b n l h'), 
            value=rearrange(v, 'b l n h -> b n l h'), 
            attn_mask=mask, 
            dropout_p=0.0
        )  # (b, n, l, h)
        
        x = rearrange(x, 'b n l h -> b l (n h)')
        x = self.proj(x)  # (b, l, d)
        return x


class MLP(nn.Module):
    r"""Ref: https://github.com/Lightning-AI/lit-llama/blob/main/lit_llama/model.py"""

    def __init__(self, n_embd: int, n_head: int) -> None:
        super().__init__()
        n_hidden = int(8 * n_embd / 3)
        self.fc1 = nn.Linear(n_embd, n_hidden, bias=False)
        self.fc2 = nn.Linear(n_embd, n_hidden, bias=False)
        self.proj = nn.Linear(n_hidden, n_embd, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        r"""Causal self attention.

        Args:
            x: (b, l, d)
           
        Outputs:
            x: (b, l, d)
        """

        x = F.silu(self.fc1(x)) * self.fc2(x)
        x = self.proj(x)
        return x


def build_causal_mask(seq_len: int) -> Tensor:
    r"""Build causal mask."""

    ones = torch.ones((seq_len, seq_len), dtype=torch.bool)  # shape: (l, l)
    mask = torch.tril(ones)[None, None, :, :]  # shape: (1, 1, l, l)
    return mask