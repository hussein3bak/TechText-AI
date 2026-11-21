"""
TechText AI - Neural Language Model Package
Implements a GPT-style Decoder-Only Transformer architecture from scratch in PyTorch.
"""

from .transformer import GPT, GPTConfig
from .generator import TextGenerator

__all__ = ["GPT", "GPTConfig", "TextGenerator"]
