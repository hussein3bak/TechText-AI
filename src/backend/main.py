"""
TechText AI - FastAPI Application Server
Provides RESTful endpoints for model inference, generation history, and system status.
Serves the modern frontend portal statically.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .database import (
    init_db,
    insert_generation,
    get_history,
    delete_history_item,
    clear_all_history,
)
from .schemas import (
    GenerateRequest,
    GenerateResponse,
    HistoryItem,
    ModelInfo,
    HealthResponse,
)
from ..model.generator import TextGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Cache of loaded TextGenerators per model name
LOADED_MODELS: Dict[str, TextGenerator] = {}
# Use absolute path so the server works regardless of working directory
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "checkpoints"


def get_or_load_model(model_name: str = "tech") -> TextGenerator:
    """Retrieves cached generator or loads it from checkpoint."""
    model_name = (model_name or "tech").lower().strip()
    if model_name in LOADED_MODELS:
        return LOADED_MODELS[model_name]

    # Map aliases to paths
    model_map = {
        "tech": CHECKPOINT_DIR / "tech_gpt2_finetuned",
        "tech_gpt2": CHECKPOINT_DIR / "tech_gpt2_finetuned",
        "tech_nanogpt": CHECKPOINT_DIR / "tech_gpt_best.pt",
        "twitter": CHECKPOINT_DIR / "twitter_gpt_best.pt",
        "twitter_nanogpt": CHECKPOINT_DIR / "twitter_gpt_best.pt",
    }

    selected_path = model_map.get(model_name)
    if not selected_path or not selected_path.exists():
        # NanoGPT is the primary model — try it first
        if (CHECKPOINT_DIR / "tech_gpt_best.pt").exists():
            selected_path = CHECKPOINT_DIR / "tech_gpt_best.pt"
        elif (CHECKPOINT_DIR / "tech_gpt2_finetuned").exists():
            selected_path = CHECKPOINT_DIR / "tech_gpt2_finetuned"
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model checkpoint for '{model_name}' not ready yet. Please train models first.",
            )

    try:
        generator = TextGenerator(str(selected_path))
        LOADED_MODELS[model_name] = generator
        return generator
    except Exception as e:
        logger.error(f"Failed to load model from {selected_path}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model: {str(e)}",
        )


app = FastAPI(
    title="TechText AI API",
    description="REST API for the TechText AI Text Generation Portal",
    version="1.0.0",
)

# CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint reporting system status and active models."""
    available_ckpts = []
    if CHECKPOINT_DIR.exists():
        if (CHECKPOINT_DIR / "tech_gpt2_finetuned").exists():
            available_ckpts.append("tech_gpt2_finetuned")
        if (CHECKPOINT_DIR / "tech_gpt_best.pt").exists():
            available_ckpts.append("tech_gpt_best.pt")

    return HealthResponse(
        status="healthy",
        app_name="TechText AI Portal",
        version="1.0.0",
        device="cpu",
        models_available=available_ckpts,
    )


@app.get("/api/models", response_model=List[ModelInfo])
def list_models():
    """Returns metadata for the two primary model checkpoints."""
    models = []

    # 1. Primary Model: NanoGPT (trained from scratch)
    tech_nano = CHECKPOINT_DIR / "tech_gpt_best.pt"
    if tech_nano.exists():
        try:
            gen = get_or_load_model("tech_nanogpt")
            models.append(
                ModelInfo(
                    id="tech_nanogpt",
                    name="NanoGPT",
                    checkpoint_file="tech_gpt_best.pt",
                    dataset="Scraped Tech Articles (Char)",
                    parameters=gen.parameters,
                    vocab_size=gen.vocab_size,
                    val_loss=gen.val_loss,
                    engine_type="NanoGPT Scratch",
                    description="Custom character-level transformer built and trained entirely from scratch using PyTorch. Demonstrates core transformer mechanics and self-attention.",
                )
            )
        except Exception as e:
            logger.warning(f"Error inspecting nano model: {e}")

    # 2. Secondary Model: TechText GPT (fine-tuned GPT-2)
    hf_path = CHECKPOINT_DIR / "tech_gpt2_finetuned"
    if hf_path.exists():
        try:
            gen = get_or_load_model("tech")
            models.append(
                ModelInfo(
                    id="tech",
                    name="TechText GPT",
                    checkpoint_file="tech_gpt2_finetuned",
                    dataset="Technology & AI Articles",
                    parameters=gen.parameters,
                    vocab_size=gen.vocab_size,
                    val_loss=gen.val_loss,
                    engine_type="Fine-Tuned Domain GPT",
                    description="GPT-2 fine-tuned on technology and AI articles. Generates coherent, fluent technical prose.",
                )
            )
        except Exception as e:
            logger.warning(f"Error inspecting tech model: {e}")

    return models


@app.post("/api/generate", response_model=GenerateResponse)
def generate_text(req: GenerateRequest):
    """Generates text from a seed prompt using the requested model."""
    generator = get_or_load_model(req.model_name or "tech")

    try:
        result = generator.generate(
            prompt=req.prompt,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
        )

        # Record to SQLite history
        record_id = insert_generation(
            prompt=result["prompt"],
            continuation=result["continuation"],
            full_text=result["full_text"],
            model_name=req.model_name or "tech",
            temperature=req.temperature,
            top_k=req.top_k,
            tokens_generated=result["tokens_generated"],
        )

        # Fetch inserted record to get timestamp
        history = get_history(limit=1)
        created_at = history[0]["created_at"] if history else ""

        return GenerateResponse(
            id=record_id,
            prompt=result["prompt"],
            continuation=result["continuation"],
            full_text=result["full_text"],
            model_name=req.model_name or "tech",
            tokens_generated=result["tokens_generated"],
            temperature=req.temperature,
            top_k=req.top_k,
            created_at=created_at,
        )
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {str(e)}",
        )


@app.get("/api/history", response_model=List[HistoryItem])
def fetch_history(limit: int = 50):
    """Fetches generation history from SQLite."""
    records = get_history(limit=limit)
    return [HistoryItem(**r) for r in records]


@app.delete("/api/history/{item_id}")
def delete_history(item_id: int):
    """Deletes a generation item by ID."""
    success = delete_history_item(item_id)
    if not success:
        raise HTTPException(status_code=404, detail="History item not found")
    return {"status": "deleted", "id": item_id}


@app.delete("/api/history")
def clear_history():
    """Clears all generation history records."""
    count = clear_all_history()
    return {"status": "cleared", "records_removed": count}


# Resolve frontend dir using absolute path
FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"


# Frontend Web Routes
@app.get("/", include_in_schema=False)
def get_frontend_index():
    """Serves the main landing page and text generation dashboard."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "TechText AI API is running. Frontend not found."}


@app.get("/style.css", include_in_schema=False)
def get_frontend_css():
    css_file = FRONTEND_DIR / "style.css"
    if css_file.exists():
        return FileResponse(css_file, media_type="text/css")
    raise HTTPException(status_code=404, detail="CSS file not found")


@app.get("/app.js", include_in_schema=False)
def get_frontend_js():
    js_file = FRONTEND_DIR / "app.js"
    if js_file.exists():
        return FileResponse(js_file, media_type="application/javascript")
    raise HTTPException(status_code=404, detail="JS file not found")

