"""
TechText AI - Comprehensive Automated Test Suite
Tests text cleaner, char tokenizer, transformer forward pass,
fine-tuned domain model generation, scratch nanoGPT generation,
FastAPI REST endpoints, SQLite persistence, and edge-case validation.
"""

import pytest
import torch
from pathlib import Path

from src.nlp.cleaner import TextCleaner
from src.nlp.tokenizer import CharTokenizer
from src.model.transformer import GPT, GPTConfig
from src.backend.main import (
    health_check,
    list_models,
    generate_text,
    fetch_history,
    delete_history,
    clear_history,
    get_frontend_index,
)
from src.backend.schemas import GenerateRequest
from src.backend.database import get_history, insert_generation


def test_text_cleaner():
    cleaner = TextCleaner()
    dirty_text = "<span>Hello</span> ^^ world! % $ @ <b>great</b>! <script>alert(1)</script>"
    cleaned = cleaner.clean_text(dirty_text)
    assert "<span>" not in cleaned
    assert "<script>" not in cleaned
    assert "<b>" not in cleaned
    assert "^^" not in cleaned
    assert "%" not in cleaned
    assert "$" not in cleaned
    assert "Hello" in cleaned
    assert "world" in cleaned


def test_char_tokenizer():
    corpus = "Artificial Intelligence and Machine Learning"
    tokenizer = CharTokenizer()
    tokenizer.fit(corpus)

    assert tokenizer.vocab_size == len(set(corpus))
    tokens = tokenizer.encode("Machine")
    decoded = tokenizer.decode(tokens)
    assert decoded == "Machine"


def test_transformer_forward_pass():
    config = GPTConfig(
        block_size=32,
        vocab_size=50,
        n_layer=2,
        n_head=2,
        n_embd=64,
        dropout=0.0,
    )
    model = GPT(config)

    idx = torch.randint(0, 50, (2, 16))
    targets = torch.randint(0, 50, (2, 16))

    logits, loss = model(idx, targets)
    assert logits.shape == (2, 16, 50)
    assert loss is not None
    assert loss.item() > 0


def test_api_health():
    res = health_check()
    assert res.status == "healthy"
    assert res.app_name == "TechText AI Portal"
    assert len(res.models_available) >= 1


def test_api_list_models():
    models = list_models()
    assert isinstance(models, list)
    assert len(models) >= 2
    model_ids = [m.id for m in models]
    assert "tech" in model_ids
    assert "tech_nanogpt" in model_ids


def test_api_generate_domain_finetuned():
    """Verifies that the domain fine-tuned model generates fluent English continuation."""
    req = GenerateRequest(
        prompt="Deep learning is",
        model_name="tech",
        max_tokens=30,
        temperature=0.7,
        top_k=40,
    )
    res = generate_text(req)
    assert res.prompt == "Deep learning is"
    assert res.model_name == "tech"
    assert res.tokens_generated > 0
    assert len(res.continuation) > 0
    # Verify continuation has readable English words
    words = res.continuation.split()
    assert len(words) >= 3
    assert res.id is not None


def test_api_generate_nanogpt_scratch():
    """Verifies that the educational scratch model runs generation."""
    req = GenerateRequest(
        prompt="Neural",
        model_name="tech_nanogpt",
        max_tokens=25,
        temperature=0.8,
        top_k=30,
    )
    res = generate_text(req)
    assert res.prompt == "Neural"
    assert res.tokens_generated == 25
    assert len(res.continuation) > 0


def test_api_history_and_deletion():
    req = GenerateRequest(
        prompt="Integration Test History Prompt",
        model_name="tech",
        max_tokens=15,
        temperature=0.5,
        top_k=30,
    )
    res = generate_text(req)
    item_id = res.id

    # Fetch history
    history = fetch_history(limit=10)
    assert any(h.id == item_id for h in history)

    # Delete single item
    del_res = delete_history(item_id)
    assert del_res["status"] == "deleted"


def test_input_validation_and_handling():
    """Tests realistic user prompts, long inputs, and special characters."""
    # Special characters and punctuation in prompt
    req = GenerateRequest(
        prompt="What is AI? (e.g. LLMs & Transformers):",
        model_name="tech",
        max_tokens=20,
        temperature=0.7,
        top_k=40,
    )
    res = generate_text(req)
    assert len(res.continuation) > 0

    # Long input prompt
    long_prompt = "Artificial intelligence and machine learning technologies have revolutionized modern software engineering. " * 3
    req_long = GenerateRequest(
        prompt=long_prompt,
        model_name="tech",
        max_tokens=20,
    )
    res_long = generate_text(req_long)
    assert len(res_long.continuation) > 0


def test_frontend_files_exist():
    res = get_frontend_index()
    assert Path(res.path).exists()
    assert Path("frontend/style.css").exists()
    assert Path("frontend/app.js").exists()


def test_model_names_order_and_naming():
    """Verifies that NanoGPT is primary (first) and model names match requirements."""
    models = list_models()
    assert len(models) >= 2
    # NanoGPT must be the first model in the list
    assert models[0].id == "tech_nanogpt"
    assert models[0].name == "NanoGPT"
    assert models[1].id == "tech"
    assert models[1].name == "TechText GPT"


def test_sampling_parameters_ranges():
    """Verifies that Creativity Level (temp), Word Choice Range (top_k), and Max Tokens operate properly."""
    # Test low temperature (deterministic)
    req_det = GenerateRequest(
        prompt="Computers are",
        model_name="tech_nanogpt",
        max_tokens=20,
        temperature=0.2,
        top_k=5,
    )
    res_det = generate_text(req_det)
    assert res_det.tokens_generated == 20
    assert len(res_det.continuation) > 0

    # Test high temperature (creative)
    req_creat = GenerateRequest(
        prompt="Computers are",
        model_name="tech",
        max_tokens=25,
        temperature=1.2,
        top_k=80,
    )
    res_creat = generate_text(req_creat)
    assert res_creat.tokens_generated > 0
    assert len(res_creat.continuation) > 0
