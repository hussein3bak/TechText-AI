"""
TechText AI - Tokenizers
Supports both Character-Level (nanoGPT-style, lightweight for CPU training)
and Byte-Pair Encoding (BPE via TikToken for high-capacity language modeling).
"""

import pickle
from pathlib import Path
from typing import List, Dict, Union, Optional
import tiktoken


class CharTokenizer:
    """
    Character-level tokenizer identical to Andrej Karpathy's nanoGPT prepare.py.
    Maps each unique character in the corpus to an integer index.
    Fast, deterministic, and ideal for learning transformer mechanics on CPU.
    """

    def __init__(self, vocab: Optional[List[str]] = None):
        self.chars: List[str] = vocab or []
        self.vocab_size: int = len(self.chars)
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(self.chars)}
        self.itos: Dict[int, str] = {i: ch for i, ch in enumerate(self.chars)}

    def fit(self, text: str):
        """Builds the character vocabulary from input text."""
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text: str) -> List[int]:
        """Encodes string to list of token IDs. Skips unseen characters with fallback."""
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, tokens: List[int]) -> str:
        """Decodes list of token IDs back into string."""
        return "".join([self.itos.get(i, "") for i in tokens])

    def save(self, filepath: Union[str, Path]):
        """Saves vocabulary mapping to a pickle metadata file."""
        meta = {
            "vocab_size": self.vocab_size,
            "itos": self.itos,
            "stoi": self.stoi,
            "chars": self.chars,
            "type": "char",
        }
        with open(filepath, "wb") as f:
            pickle.dump(meta, f)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "CharTokenizer":
        """Loads tokenizer from a pickle metadata file."""
        with open(filepath, "rb") as f:
            meta = pickle.load(f)
        tok = cls(vocab=meta.get("chars", []))
        tok.vocab_size = meta["vocab_size"]
        tok.stoi = meta["stoi"]
        tok.itos = meta["itos"]
        return tok


class BPETokenizer:
    """
    Subword tokenizer using OpenAI's TikToken (GPT-2 encoding).
    Compresses text into subwords, giving higher semantic density per token.
    """

    def __init__(self, encoding_name: str = "gpt2"):
        self.encoding_name = encoding_name
        self.enc = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.enc.n_vocab

    def encode(self, text: str) -> List[int]:
        return self.enc.encode_ordinary(text)

    def decode(self, tokens: List[int]) -> str:
        return self.enc.decode(tokens)

    def save(self, filepath: Union[str, Path]):
        meta = {
            "vocab_size": self.vocab_size,
            "encoding_name": self.encoding_name,
            "type": "bpe",
        }
        with open(filepath, "wb") as f:
            pickle.dump(meta, f)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "BPETokenizer":
        with open(filepath, "rb") as f:
            meta = pickle.load(f)
        return cls(encoding_name=meta.get("encoding_name", "gpt2"))
