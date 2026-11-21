"""
TechText AI - Pydantic Request & Response Schemas
Validates API payloads for text generation, history management, and health status.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000, description="Seed prompt for text generation")
    model_name: Optional[str] = Field("tech", description="Model checkpoint identifier (e.g. 'tech', 'twitter')")
    max_tokens: int = Field(120, ge=10, le=500, description="Maximum new tokens to generate")
    temperature: float = Field(0.8, ge=0.0, le=2.0, description="Sampling randomness (0=deterministic/greedy)")
    top_k: int = Field(40, ge=1, le=200, description="Top-K sampling cutoff")
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0, description="Penalty for repeating characters/tokens")


class GenerateResponse(BaseModel):
    id: Optional[int] = None
    prompt: str
    continuation: str
    full_text: str
    model_name: str
    tokens_generated: int
    temperature: float
    top_k: int
    created_at: str


class HistoryItem(BaseModel):
    id: int
    prompt: str
    continuation: str
    full_text: str
    model_name: str
    temperature: float
    top_k: int
    tokens_generated: int
    created_at: str


class ModelInfo(BaseModel):
    id: str
    name: str
    checkpoint_file: str
    dataset: str
    parameters: int
    vocab_size: int
    val_loss: Optional[float] = None
    engine_type: Optional[str] = "Fine-Tuned Domain GPT"
    description: Optional[str] = None



class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    device: str
    models_available: List[str]
