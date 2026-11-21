"""
TechText AI - GPT-Style Decoder-Only Transformer Architecture
Built from scratch in PyTorch following the architecture of Radford et al. (GPT-2)
and Andrej Karpathy's nanoGPT.

Components:
1. CausalSelfAttention: Multi-Head Self-Attention with causal autoregressive masking.
2. MLP: Feed-Forward network with GELU non-linearity.
3. Block: Transformer block with Pre-LayerNorm and residual connections.
4. GPT: Full model with token/position embeddings, transformer blocks, and language model head.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 128      # Maximum context sequence length (time steps)
    vocab_size: int = 65       # Total number of unique tokens in vocabulary
    n_layer: int = 4           # Number of transformer layers/blocks
    n_head: int = 4            # Number of attention heads
    n_embd: int = 128          # Embedding dimension
    dropout: float = 0.1       # Dropout probability
    bias: bool = True          # Use bias in Linears and LayerNorms


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention mechanism.
    Ensures that prediction for position i depends only on tokens at indices <= i.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"

        # Key, Query, Value projections for all heads packed into a single linear layer
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # Causal mask to ensure attention is only directed to previous positions
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # Batch size, Sequence length, Embedding dimension

        # Calculate Query, Key, Values for all heads in batch and move head forward to the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # Apply causal mask (mask out upper triangular future positions)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # Re-assemble all head outputs side by side

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Feed-Forward Neural Network with GELU activation and 4x channel expansion."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Standard Transformer Block: Pre-LayerNorm followed by Self-Attention and MLP."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, elementwise_affine=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, elementwise_affine=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connections: x = x + Attention(LN(x))
        x = x + self.attn(self.ln_1(x))
        # Residual connections: x = x + MLP(LN(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """The full GPT Language Model."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd, elementwise_affine=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: tie weights between token embedding and final projection head
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights with standard normal distribution
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # Report number of parameters
        n_params = sum(p.numel() for p in self.parameters())
        logger_name = f"GPT Model initialized with {n_params/1e6:.2f}M parameters ({n_params:,} total)"
        self.num_parameters = n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is {self.config.block_size}"

        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # Forward token and position embeddings
        tok_emb = self.transformer.wte(idx)  # (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Forward through all transformer blocks
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # If targets are provided, calculate cross-entropy loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference optimization: forward only the last token position
            logits = self.lm_head(x[:, [-1], :])  # (b, 1, vocab_size)
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        repetition_penalty: float = 1.1,
    ) -> torch.Tensor:
        """
        Generates autoregressively token by token conditioned on prompt idx.
        """
        for _ in range(max_new_tokens):
            # Crop context if it exceeds block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]

            logits, _ = self(idx_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :]  # (B, vocab_size)

            # Apply repetition penalty if specified (penalize immediate repetitive loops)
            if repetition_penalty > 1.0:
                for b_idx in range(idx.shape[0]):
                    # Look at recent window (last 20 tokens) to avoid penalizing common vowels globally
                    recent_tokens = idx[b_idx, -min(20, idx.shape[1]):]
                    unique_tokens = torch.unique(recent_tokens)
                    # Only penalize tokens that appeared multiple times recently
                    for ut in unique_tokens:
                        count = (recent_tokens == ut).sum()
                        if count > 2:
                            logits[b_idx, ut] /= (repetition_penalty ** (count.item() - 1))

            # Scale by temperature
            if temperature == 0.0:
                # Greedy decoding (argmax)
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / max(temperature, 1e-5)

                # Optional top-k filtering
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            # Append sampled index to sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
