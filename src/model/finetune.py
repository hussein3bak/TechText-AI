"""
TechText AI - Domain Fine-Tuning Pipeline
Fine-tunes a local PyTorch GPT-2/DistilGPT-2 model on our scraped technology corpus.
Produces grammatically sound, coherent, domain-adapted AI generation without external APIs.
"""

import os
import json
import time
import argparse
import logging
from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class TextBlockDataset(Dataset):
    """Chunks tokenized text into fixed-length blocks for autoregressive training."""

    def __init__(self, token_ids: List[int], block_size: int = 128):
        self.block_size = block_size
        # Slice into chunks of block_size
        n_blocks = len(token_ids) // block_size
        self.examples = [
            torch.tensor(token_ids[i * block_size : (i + 1) * block_size], dtype=torch.long)
            for i in range(n_blocks)
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        return {"input_ids": item, "labels": item.clone()}


def train_domain_model(
    corpus_path: str = "data/processed/tech_cleaned.txt",
    output_dir: str = "models/checkpoints/tech_gpt2_finetuned",
    base_model_name: str = "distilgpt2",
    block_size: int = 128,
    batch_size: int = 4,
    epochs: int = 4,
    lr: float = 5e-5,
    device: str = "cpu",
) -> Dict:
    """
    Fine-tunes the base model on our technology dataset.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    corpus_file = Path(corpus_path)

    if not corpus_file.exists():
        raise FileNotFoundError(f"Cleaned corpus not found at {corpus_file}. Run preprocessor first!")

    logger.info(f"Loading corpus from: {corpus_file}")
    with open(corpus_file, "r", encoding="utf-8") as f:
        text = f.read()

    logger.info(f"Loaded {len(text):,} characters for domain fine-tuning.")

    # 1. Load Tokenizer & Model (prefer local cache)
    logger.info(f"Loading base model: {base_model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(base_model_name, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        model = AutoModelForCausalLM.from_pretrained(base_model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)

    # 2. Tokenize corpus
    token_ids = tokenizer.encode(text)
    logger.info(f"Corpus tokenized into {len(token_ids):,} BPE tokens.")

    dataset = TextBlockDataset(token_ids, block_size=block_size)
    if len(dataset) == 0:
        raise ValueError("Corpus too small to create training blocks!")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    logger.info(f"Created {len(dataset)} training blocks of size {block_size}.")

    # 3. Setup Optimizer
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # 4. Training Loop
    model.train()
    total_steps = len(loader) * epochs
    logger.info(f"Starting fine-tuning: {epochs} epochs ({total_steps} total steps) on {device}...")

    loss_history = []
    start_time = time.time()
    step_count = 0
    initial_loss = None

    for epoch in range(1, epochs + 1):
        epoch_losses = []
        for batch in loader:
            step_count += 1
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            if initial_loss is None:
                initial_loss = float(loss.item())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_val = float(loss.item())
            epoch_losses.append(loss_val)

            if step_count % 15 == 0 or step_count == total_steps:
                logger.info(
                    f"Epoch {epoch}/{epochs} | Step {step_count}/{total_steps} | Loss: {loss_val:.4f}"
                )

        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        loss_history.append(avg_epoch_loss)
        logger.info(f"--- Completed Epoch {epoch}/{epochs} | Average Loss: {avg_epoch_loss:.4f} ---")

    training_time = time.time() - start_time
    final_loss = loss_history[-1] if loss_history else initial_loss

    # 5. Save fine-tuned checkpoint & tokenizer
    logger.info(f"Saving fine-tuned model checkpoint to {output_path}...")
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    metrics = {
        "base_model": base_model_name,
        "parameters": sum(p.numel() for p in model.parameters()),
        "dataset": str(corpus_file),
        "total_tokens": len(token_ids),
        "initial_loss": round(float(initial_loss), 4),
        "final_loss": round(float(final_loss), 4),
        "epochs": epochs,
        "training_time_seconds": round(training_time, 2),
        "device": device,
    }

    with open(output_path / "finetune_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info(
        f"Domain fine-tuning complete in {training_time:.1f}s! Initial Loss: {initial_loss:.4f} -> Final Loss: {final_loss:.4f}"
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Fine-tune TechText GPT on Technology Dataset")
    parser.add_argument("--corpus", type=str, default="data/processed/tech_cleaned.txt")
    parser.add_argument("--output_dir", type=str, default="models/checkpoints/tech_gpt2_finetuned")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    train_domain_model(
        corpus_path=args.corpus,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )


if __name__ == "__main__":
    main()
