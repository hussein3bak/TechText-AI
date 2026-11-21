"""
TechText AI - Language Model Training Pipeline
Trains our custom GPT Transformer on technology or conversational datasets.
Saves model checkpoints, tracks training and validation loss, and logs metrics.
"""

import os
import time
import math
import pickle
import argparse
import logging
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import torch
from torch.optim import AdamW

from .transformer import GPT, GPTConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        dataset: str = "tech",
        processed_dir: str = "data/processed",
        checkpoint_dir: str = "models/checkpoints",
        batch_size: int = 16,
        block_size: int = 128,
        n_layer: int = 4,
        n_head: int = 4,
        n_embd: int = 128,
        dropout: float = 0.1,
        learning_rate: float = 1.5e-3,
        max_iters: int = 1200,
        eval_interval: int = 150,
        eval_iters: int = 5,
        device: str = "cpu",
    ):
        self.dataset = dataset
        self.dataset_dir = Path(processed_dir) / dataset
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.batch_size = batch_size
        self.block_size = block_size
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.eval_interval = eval_interval
        self.eval_iters = eval_iters

        # Select computing device
        if device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        logger.info(f"Using device for training: {self.device}")

        # Load dataset binaries
        self._load_data()

        # Initialize model
        self.config = GPTConfig(
            block_size=self.block_size,
            vocab_size=self.vocab_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            dropout=dropout,
        )
        self.model = GPT(self.config).to(self.device)
        logger.info(f"Initialized TechText GPT with {self.model.num_parameters:,} parameters")

        # Initialize Optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-2)

    def _load_data(self):
        train_bin = self.dataset_dir / "train.bin"
        val_bin = self.dataset_dir / "val.bin"
        meta_pkl = self.dataset_dir / "meta.pkl"

        if not train_bin.exists() or not meta_pkl.exists():
            raise FileNotFoundError(
                f"Dataset binaries missing at {self.dataset_dir}. Run src.nlp.preprocessor first!"
            )

        self.train_data = np.fromfile(train_bin, dtype=np.uint16)
        self.val_data = (
            np.fromfile(val_bin, dtype=np.uint16) if val_bin.exists() else self.train_data
        )

        with open(meta_pkl, "rb") as f:
            self.meta = pickle.load(f)
        self.vocab_size = self.meta["vocab_size"]
        self.itos = self.meta.get("itos", {})
        self.stoi = self.meta.get("stoi", {})

        logger.info(
            f"Loaded dataset '{self.dataset}': Train tokens={len(self.train_data):,}, "
            f"Val tokens={len(self.val_data):,}, Vocab size={self.vocab_size}"
        )

    def get_batch(self, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates a batch of inputs x and targets y using vectorized indexing."""
        data = self.train_data if split == "train" else self.val_data
        max_idx = len(data) - self.block_size - 1
        if max_idx <= 0:
            raise ValueError(f"Dataset too small ({len(data)} tokens) for block_size {self.block_size}")

        ix = np.random.randint(0, max_idx, size=(self.batch_size,))
        offsets = np.arange(self.block_size)
        indices = ix[:, None] + offsets[None, :]
        x = torch.from_numpy(data[indices].astype(np.int64))
        y = torch.from_numpy(data[indices + 1].astype(np.int64))
        return x.to(self.device), y.to(self.device)

    @torch.no_grad()
    def estimate_loss(self) -> Dict[str, float]:
        """Evaluates average loss on train and validation splits."""
        out = {}
        self.model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                x, y = self.get_batch(split)
                _, loss = self.model(x, y)
                losses[k] = loss.item()
            out[split] = float(losses.mean())
        self.model.train()
        return out

    def get_lr(self, it: int) -> float:
        """Cosine annealing learning rate schedule with warmup."""
        warmup_iters = min(50, max(1, self.max_iters // 10))
        lr_decay_iters = self.max_iters
        min_lr = self.learning_rate * 0.1

        if it < warmup_iters:
            return self.learning_rate * (it + 1) / (warmup_iters + 1)
        if it > lr_decay_iters:
            return min_lr
        if lr_decay_iters <= warmup_iters:
            return min_lr
        decay_ratio = (it - warmup_iters) / max(1, lr_decay_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (self.learning_rate - min_lr)

    def save_checkpoint(self, filename: str, iteration: int, best_val_loss: float):
        """Saves weights, config, and vocabulary metadata."""
        ckpt_path = self.checkpoint_dir / filename
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "config": self.config.__dict__,
            "meta": self.meta,
            "iteration": iteration,
            "best_val_loss": best_val_loss,
            "dataset": self.dataset,
        }
        torch.save(checkpoint, ckpt_path)
        logger.info(f" Saved checkpoint to {ckpt_path} (Val Loss: {best_val_loss:.4f})")

    def train(self):
        """Executes the training loop."""
        logger.info(f"Starting training run: {self.max_iters} iterations...")
        best_val_loss = float("inf")
        start_time = time.time()

        for iter_num in range(1, self.max_iters + 1):
            # Update learning rate
            lr = self.get_lr(iter_num)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # Evaluate periodically
            if iter_num % self.eval_interval == 0 or iter_num == self.max_iters:
                losses = self.estimate_loss()
                elapsed = time.time() - start_time
                logger.info(
                    f"Step {iter_num:4d}/{self.max_iters} | "
                    f"Train Loss: {losses['train']:.4f} | "
                    f"Val Loss: {losses['val']:.4f} | "
                    f"LR: {lr:.6f} | "
                    f"Elapsed: {elapsed:.1f}s"
                )

                # Save best checkpoint
                if losses["val"] < best_val_loss:
                    best_val_loss = losses["val"]
                    self.save_checkpoint(f"{self.dataset}_gpt_best.pt", iter_num, best_val_loss)

            # Sample batch and perform forward/backward pass
            x, y = self.get_batch("train")
            logits, loss = self.model(x, y)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

        # Save final checkpoint
        self.save_checkpoint(f"{self.dataset}_gpt_final.pt", self.max_iters, best_val_loss)
        total_time = time.time() - start_time
        logger.info(f"Training completed successfully in {total_time:.1f}s! Best Val Loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train TechText GPT Model")
    parser.add_argument("--dataset", type=str, default="tech", choices=["tech", "twitter", "movies"])
    parser.add_argument("--max_iters", type=int, default=1200, help="Training iterations")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--block_size", type=int, default=128, help="Context length")
    parser.add_argument("--n_layer", type=int, default=4, help="Transformer layers")
    parser.add_argument("--n_head", type=int, default=4, help="Attention heads")
    parser.add_argument("--n_embd", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--lr", type=float, default=1.5e-3, help="Learning rate")
    parser.add_argument("--eval_interval", type=int, default=150, help="Eval interval")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    args = parser.parse_args()

    trainer = Trainer(
        dataset=args.dataset,
        max_iters=args.max_iters,
        batch_size=args.batch_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        learning_rate=args.lr,
        eval_interval=args.eval_interval,
        device=args.device,
    )
    trainer.train()


if __name__ == "__main__":
    main()
