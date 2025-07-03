from __future__ import annotations
import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


class GPT2(nn.Module):
    def __init__(self, config: GPTConfig):
        r"""GPT2. Ref: https://github.com/karpathy/nanoGPT/blob/master/model.py"""

        super().__init__()
        
        self.config = config

        # Word to embedding (wte) and word position embedding (wpe)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # Output layers
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Bind weights
        self.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)

        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, ids: torch.LongTensor) -> torch.LongTensor:
        r"""Next id prediction with GPT2.

        b: batch_size
        t: time_steps
        d: hidden_size
        v: vocab_size

        Args:
            ids: (b, t)

        Outputs:
            logits: (b, t, v)
        """

        device = ids.device
        B, T = ids.shape

        assert T <= self.config.block_size, "Can not forward sequence of {T} > {self.config.block_size}"

        # Absolute positions
        pos = torch.arange(0, T, dtype=torch.long, device=device)  # shape: (t,)

        # ID embedding and position embedding
        id_emb = self.wte(ids)  # shape: (b, t, d)
        pos_emb = self.wpe(pos)  # shape: (t, d)
        x = self.drop(id_emb + pos_emb)  # shape; (b, t, d)

        # Transformer
        for block in self.blocks:
            x = block(x)
        # x: (b, t, d)

        # Output layers
        x = self.ln_f(x)  # shape: (b, t, d)
        logits = self.lm_head(x)  # shape: (b, t, v)

        return logits

    @torch.no_grad()
    def generate(
        self, 
        ids: torch.LongTensor, 
        max_new_ids: int, 
        temperature: float = 1.0, 
        top_k: None | int = None
    ):
        r"""Next ID sampling with auto-regression. Make sure to use model.eval()

        b: batch_size
        t: time_steps
        v: vocab_size

        Args:
            ids: (b, 1)
            max_new_ids: int
            temperature: float
            top_k: None | int

        Returns:
            new_ids: (b, t), sampled IDs
        """
        input_len = ids.shape[1]

        for _ in range(max_new_ids):

            # If the sequence context is growing too long we must crop it at block_size
            if ids.shape[1] <= self.config.block_size:
                prev_ids = ids
            else:
                prev_ids = ids[:, -self.config.block_size:]

            # Forward
            logits = self(prev_ids)  # shape: (b, t, v)

            # Take the final step logits
            logits = logits[:, -1, :] / temperature  # shape: (b, v)

            # Crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)  # shape: (b, v)

            # Sample the next ID
            next_id = torch.multinomial(probs, num_samples=1)  # shape: (b, 1)

            # Append the sampled ID to the running IDs and continue
            ids = torch.cat((ids, next_id), dim=1)  # shape: (b, t)

        new_ids = ids[:, input_len:]  # shape: (b, t)

        return new_ids


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()

        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        r"""Causal self attention.

        b: batch size
        t: time steps
        d: latent dim
        h: heads num

        Args:
            x: (b, t, d)

        Outputs:
            x: (b, t, d)
        """

        B, T, D = x.shape

        # Calculate query, key, values
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # q, k, v shapes: (b, t, d)

        k = k.view(B, T, self.n_head, D // self.n_head).transpose(1, 2)  
        q = q.view(B, T, self.n_head, D // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, D // self.n_head).transpose(1, 2)
        # q, k, v shapes: (b, t, h, d/h)

        # Causal self-attention
        if self.flash:
            # Efficient attention using Flash Attention CUDA kernels
            x = torch.nn.functional.scaled_dot_product_attention(
                query=q, 
                key=k, 
                value=v, 
                attn_mask=None, 
                dropout_p=self.dropout if self.training else 0, 
                is_causal=True
            )
            # shape: (b, h, t, d/h)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # shape: (b, h, t, t)
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))  # shape: (b, h, t, t)
            att = F.softmax(att, dim=-1)  # shape: (b, h, t, t)
            att = self.attn_dropout(att)  # shape: (b, h, t, t)
            x = att @ v  # shape: (b, h, t, d/h)

        x = x.transpose(1, 2).contiguous().view(B, T, D)  # shape: (b, t, d)

        # output projection
        x = self.resid_dropout(self.c_proj(x))  # shape: (b, t, d)
        
        return x

class MLP(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        r"""MLP.

        Args:
            x: (b, t, d)

        Outputs:
            x: (b, t, d)
        """
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.att_norm = LayerNorm(config.n_embd, bias=config.bias)
        self.att = CausalSelfAttention(config)
        self.ffn_norm = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        r"""MLP.

        Args:
            x: (b, t, d)

        Outputs:
            x: (b, t, d)
        """
        x = x + self.att(self.att_norm(x))
        x = x + self.mlp(self.ffn_norm(x))
        return x