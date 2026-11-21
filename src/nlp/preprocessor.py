"""
TechText AI - Complete NLP Preprocessing Pipeline
Reads raw text files, cleans noise/HTML/corruption, creates token mappings,
splits into train/val datasets, and exports binary data arrays (.bin) and metadata (meta.pkl).
"""

import os
import argparse
import logging
from pathlib import Path
import numpy as np

from .cleaner import TextCleaner
from .tokenizer import CharTokenizer, BPETokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class DatasetPreprocessor:
    def __init__(
        self,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
        tokenizer_type: str = "char",
        val_split_ratio: float = 0.1,
    ):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer_type = tokenizer_type.lower()
        self.val_split_ratio = val_split_ratio
        self.cleaner = TextCleaner()

    def process(self, input_filename: str, output_name: str):
        """
        Cleans and tokenizes a dataset file.
        Writes:
          data/processed/{output_name}_cleaned.txt
          data/processed/{output_name}/train.bin
          data/processed/{output_name}/val.bin
          data/processed/{output_name}/meta.pkl
        """
        input_path = self.raw_dir / input_filename
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found at: {input_path}")

        logger.info(f"Loading raw dataset from {input_path}...")
        with open(input_path, "r", encoding="utf-8", errors="replace") as f:
            raw_text = f.read()

        logger.info(f"Raw text length: {len(raw_text):,} characters")

        # Step 1: Clean
        logger.info("Cleaning raw text (stripping HTML, noise tokens, formatting)...")
        cleaned_text = self.cleaner.clean_text(raw_text)
        logger.info(f"Cleaned text length: {len(cleaned_text):,} characters")

        # Save cleaned text for inspection
        cleaned_txt_path = self.processed_dir / f"{output_name}_cleaned.txt"
        with open(cleaned_txt_path, "w", encoding="utf-8") as f:
            f.write(cleaned_text)
        logger.info(f"Saved human-readable cleaned text to {cleaned_txt_path}")

        # Step 2: Initialize Tokenizer
        dataset_folder = self.processed_dir / output_name
        dataset_folder.mkdir(parents=True, exist_ok=True)

        if self.tokenizer_type == "char":
            tokenizer = CharTokenizer()
            tokenizer.fit(cleaned_text)
            logger.info(f"CharTokenizer fitted. Unique characters (vocab size): {tokenizer.vocab_size}")
            logger.info(f"Vocab sample: {''.join(tokenizer.chars[:30])}...")
        elif self.tokenizer_type == "bpe":
            tokenizer = BPETokenizer()
            logger.info(f"BPETokenizer loaded. Vocab size: {tokenizer.vocab_size}")
        else:
            raise ValueError(f"Unknown tokenizer type: {self.tokenizer_type}")

        # Step 3: Train / Val Split
        n = len(cleaned_text)
        train_cutoff = int(n * (1.0 - self.val_split_ratio))
        train_text = cleaned_text[:train_cutoff]
        val_text = cleaned_text[train_cutoff:]

        logger.info(f"Splitting data: Train ({len(train_text):,} chars), Val ({len(val_text):,} chars)")

        # Step 4: Tokenize & Convert to numpy arrays
        train_ids = np.array(tokenizer.encode(train_text), dtype=np.uint16)
        val_ids = np.array(tokenizer.encode(val_text), dtype=np.uint16)

        logger.info(f"Train has {len(train_ids):,} tokens")
        logger.info(f"Val has {len(val_ids):,} tokens")

        # Step 5: Export binary files & metadata
        train_bin_path = dataset_folder / "train.bin"
        val_bin_path = dataset_folder / "val.bin"
        meta_path = dataset_folder / "meta.pkl"

        train_ids.tofile(train_bin_path)
        val_ids.tofile(val_bin_path)
        tokenizer.save(meta_path)

        logger.info(f"Exported train binary: {train_bin_path}")
        logger.info(f"Exported val binary:   {val_bin_path}")
        logger.info(f"Exported metadata:     {meta_path}")
        logger.info("Preprocessing complete!\n")


def main():
    parser = argparse.ArgumentParser(description="TechText AI Preprocessing Pipeline")
    parser.add_argument(
        "--dataset",
        type=str,
        default="tech",
        choices=["tech", "twitter", "movies"],
        help="Dataset to process: 'tech', 'twitter', or 'movies'",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="char",
        choices=["char", "bpe"],
        help="Tokenizer type: 'char' (recommended for CPU training) or 'bpe'",
    )
    args = parser.parse_args()

    preprocessor = DatasetPreprocessor(tokenizer_type=args.tokenizer)

    dataset_map = {
        "tech": ("tech_corpus.txt", "tech"),
        "twitter": ("corrupt_twitter_corpus.txt", "twitter"),
        "movies": ("corrupt_movie_corpus.txt", "movies"),
    }

    input_file, output_name = dataset_map[args.dataset]
    preprocessor.process(input_file, output_name)


if __name__ == "__main__":
    main()
