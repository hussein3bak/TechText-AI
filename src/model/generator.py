"""
TechText AI - Unified Text Generation Engine
Supports both:
1. Fine-tuned Neural Domain Language Model (BPE subword Transformer, e.g. DistilGPT-2 fine-tuned on tech)
   -> Produces grammatically fluent, coherent, domain-relevant generation.
2. Custom Scratch nanoGPT Transformer (.pt checkpoints)
   -> Educational character-level model demonstrating raw transformer mechanics.
"""

import os
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

import torch
from .transformer import GPT, GPTConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class TextGenerator:
    def __init__(self, checkpoint_path: Union[str, Path], device: str = "cpu"):
        self.device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {self.checkpoint_path}")

        # Detect whether checkpoint is a HF Transformer folder or a PyTorch .pt file
        if self.checkpoint_path.is_dir() or (self.checkpoint_path / "config.json").exists():
            self._load_hf_model()
        else:
            self._load_nanogpt_model()

    def _load_hf_model(self):
        """Loads fine-tuned Hugging Face transformer model & tokenizer."""
        from transformers import AutoTokenizer, AutoModelForCausalLM

        logger.info(f"Loading Fine-Tuned Transformer from {self.checkpoint_path} on {self.device}...")
        self.engine_type = "hf_finetuned"
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.checkpoint_path), local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(str(self.checkpoint_path), local_files_only=True)
        self.model.to(self.device)
        self.model.eval()

        self.parameters = sum(p.numel() for p in self.model.parameters())
        self.vocab_size = len(self.tokenizer)
        self.dataset = "tech (domain fine-tuned)"
        self.val_loss = None

        # Check for metrics
        metrics_file = self.checkpoint_path / "finetune_metrics.json"
        if metrics_file.exists():
            import json
            with open(metrics_file, "r") as f:
                data = json.load(f)
                self.val_loss = data.get("final_loss")

        logger.info(f"Loaded HF model ({self.parameters:,} parameters, Vocab: {self.vocab_size:,})")

    def _load_nanogpt_model(self):
        """Loads custom scratch nanoGPT checkpoint (.pt)."""
        logger.info(f"Loading Scratch nanoGPT checkpoint from {self.checkpoint_path} on {self.device}...")
        self.engine_type = "nanogpt_scratch"
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        cfg_dict = checkpoint["config"]
        self.config = GPTConfig(**cfg_dict)
        self.model = GPT(self.config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.meta = checkpoint["meta"]
        self.stoi = self.meta["stoi"]
        self.itos = {int(k): v for k, v in self.meta["itos"].items()}
        self.dataset = checkpoint.get("dataset", "unknown")
        self.val_loss = checkpoint.get("best_val_loss", None)
        self.parameters = self.model.num_parameters
        self.vocab_size = len(self.itos)

        logger.info(f"Loaded Scratch nanoGPT ({self.parameters:,} parameters, Vocab: {self.vocab_size})")

    def generate(
        self,
        prompt: str = "",
        max_new_tokens: int = 120,
        temperature: float = 0.8,
        top_k: int = 40,
        repetition_penalty: float = 1.1,
    ) -> Dict[str, Any]:
        """Unified text generation for both engine types."""
        if not prompt.strip():
            prompt = "Technology"

        if self.engine_type == "hf_finetuned":
            return self._generate_hf(prompt, max_new_tokens, temperature, top_k, repetition_penalty)
        else:
            return self._generate_nanogpt(prompt, max_new_tokens, temperature, top_k, repetition_penalty)

    def _generate_hf(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        repetition_penalty: float,
    ) -> Dict[str, Any]:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        # Handle temperature limits
        do_sample = temperature > 0.05
        safe_temp = max(float(temperature), 0.1) if do_sample else 1.0

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=safe_temp,
                top_k=max(int(top_k), 1),
                repetition_penalty=max(float(repetition_penalty), 1.0),
                pad_token_id=self.tokenizer.eos_token_id,
            )

        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Continuation is text after prompt
        continuation = full_text[len(prompt):].strip()
        tokens_gen = outputs[0].shape[0] - input_len

        return {
            "prompt": prompt,
            "continuation": continuation,
            "full_text": full_text,
            "tokens_generated": int(tokens_gen),
            "temperature": temperature,
            "top_k": top_k,
            "model_dataset": self.dataset,
            "engine": "Fine-Tuned Domain GPT",
        }

    def _generate_nanogpt(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        repetition_penalty: float,
    ) -> Dict[str, Any]:
        token_ids = [self.stoi[c] for c in prompt if c in self.stoi]
        if len(token_ids) == 0:
            token_ids = [self.stoi.get(" ", 0)]

        x = torch.tensor(token_ids, dtype=torch.long, device=self.device).unsqueeze(0)

        # Scale top_k appropriately for character-level vocabulary
        # A character vocabulary has ~70-90 tokens. Top-k of 40 permits ~50% of the entire alphabet
        # at every step, which injects random noise symbols into words.
        # For character models, effective top_k is scaled to 3-10 tokens.
        if self.vocab_size < 150:
            eff_top_k = max(2, min(int(top_k * 0.15) if top_k > 10 else top_k, 10))
        else:
            eff_top_k = max(int(top_k), 1)

        # Calibrate temperature for character level to prevent non-word character jitter
        eff_temp = max(float(temperature), 0.05)

        with torch.no_grad():
            output_tokens = self.model.generate(
                x,
                max_new_tokens=max_new_tokens,
                temperature=eff_temp,
                top_k=eff_top_k,
                repetition_penalty=repetition_penalty,
            )

        generated_ids = output_tokens[0].tolist()
        full_text = "".join([self.itos.get(int(i), "") for i in generated_ids])
        continuation = "".join([self.itos.get(int(i), "") for i in generated_ids[len(token_ids) :]])

        return {
            "prompt": prompt,
            "continuation": continuation,
            "full_text": full_text,
            "tokens_generated": len(generated_ids) - len(token_ids),
            "temperature": temperature,
            "top_k": top_k,
            "model_dataset": self.dataset,
            "engine": "NanoGPT Character Scratch",
        }


def main():
    parser = argparse.ArgumentParser(description="Generate text with TechText GPT")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/tech_gpt2_finetuned")
    parser.add_argument("--prompt", type=str, default="Artificial intelligence is transforming")
    parser.add_argument("--max_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=40)
    args = parser.parse_args()

    generator = TextGenerator(args.checkpoint)
    res = generator.generate(
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print("\n" + "=" * 60)
    print("ENGINE:", res["engine"])
    print("PROMPT:", res["prompt"])
    print("CONTINUATION:\n", res["continuation"])
    print("=" * 60)


if __name__ == "__main__":
    main()
