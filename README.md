# TechText AI — AI-Powered Text Generation Portal

An end-to-end AI system that collects technology-related text from the web, processes it using Natural Language Processing (NLP) techniques, trains custom language models, and serves a web application for interactive AI text generation.

---

## What It Does

TechText AI is a full-stack AI application built from scratch. It covers the complete machine learning pipeline:

1. **Web Scraping** — Collects technology and AI articles from the web using BeautifulSoup, with robots.txt compliance checking.
2. **NLP Preprocessing** — Cleans raw text by removing HTML tags, normalising characters, handling tokenisation, and filtering noise.
3. **Model Training** — Trains a custom character-level GPT transformer from scratch using PyTorch.
4. **Model Fine-Tuning** — Fine-tunes a GPT-2 base model on the collected technology corpus using HuggingFace Transformers.
5. **Web Application** — Serves both models through a FastAPI backend and a modern web portal where users can type a prompt and receive AI-generated text in real time.
6. **Generation History** — All generations are stored in a SQLite database and viewable in the History tab.

---

## Interactive Sampling Controls

The portal provides an intuitive, user-friendly controls panel for tuning generation behavior:

| UI Control | Technical Parameter | Description |
|------------|---------------------|-------------|
| **Creativity Level** | `Temperature` (0.10 – 1.50) | Controls how predictable or creative the generated text is. Lower values produce deterministic output; higher values increase diversity. |
| **Word Choice Range** | `Top-K` (1 – 100) | Controls how many of the most likely next-token choices the AI considers at each step, cutting off low-probability candidates. |
| **Maximum Output Length** | `Tokens to Generate` (30 – 300) | Controls the target maximum length of text the AI generates. |

---

## Models

| Model | Description |
|-------|-------------|
| **NanoGPT** *(Primary)* | A character-level transformer built entirely from scratch with PyTorch (~821K parameters, 4 layers, 4 attention heads, 128 embedding dimensions). Trained on a curated ~855K-character corpus of technology and AI literature. Demonstrates pure self-attention and transformer mechanics. |
| **TechText GPT** | GPT-2 fine-tuned on scraped technology and AI articles (~81.9M parameters). Produces fluent, coherent technical prose. |

---

## Technologies Used

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.11, FastAPI, Uvicorn |
| AI / ML | PyTorch, HuggingFace Transformers |
| NLP | BeautifulSoup, custom cleaner and tokeniser |
| Database | SQLite (via Python `sqlite3`) |
| Frontend | HTML, CSS, JavaScript (vanilla) |
| Packaging | `uv` (virtual environment + dependency management) |
| Containerisation | Docker, Docker Compose |

---

## Project Structure

```
TechText-AI/
├── presentation/
│   └── TechText_AI_Presentation.pptx   ← Project presentation slides
│
├── frontend/                            ← Web portal (HTML, CSS, JS)
├── src/
│   ├── backend/                         ← FastAPI server and SQLite database
│   ├── model/                           ← GPT transformer, training, fine-tuning
│   ├── nlp/                             ← Text cleaner, tokeniser, preprocessor
│   └── scraper/                         ← Web scraper and robots.txt checker
│
├── models/checkpoints/                  ← Model config files (weights excluded from git)
├── tests/                               ← Automated test suite (12 tests)
├── check_setup.py                       ← Environment verification script
├── requirements.txt                     ← Python dependencies
├── Dockerfile                           ← Container build file
└── docker-compose.yml                   ← Container orchestration
```

---

## Installation and Setup

### Requirements
- Python 3.11+
- `uv` package manager (recommended) — install with `pip install uv`

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/hussein3bak/TechText-AI.git
cd TechText-AI

# 2. Create virtual environment and install dependencies
uv venv
uv pip install -r requirements.txt

# 3. Verify your environment
python check_setup.py

# 4. Start the application
uvicorn src.backend.main:app --reload

# 5. Open in your browser
# http://localhost:8000
```

> **Note on model weights:** 
> - The raw training corpus (`data/raw/tech_corpus.txt`) is included in the repository so you can immediately preprocess and train the models (`python -m src.nlp.preprocessor` and `python -m src.model.train`).
> - Pre-trained model checkpoint configurations are provided in `models/checkpoints/`. Large checkpoint weights (>300MB) exceed GitHub file limits and can be generated using the training scripts or placed directly in `models/checkpoints/`.

---

## Running the Tests

```bash
python -m pytest tests/test_api.py -v
```

All 12 tests cover the NLP pipeline, the GPT transformer, both models' generation, sampling parameter controls, the API endpoints, input validation, and the frontend file structure.

---

## Configuration

Copy `.env.example` to `.env` and edit as needed:

```bash
cp .env.example .env
```

The application runs fully locally with no external API keys required.

---

## Project Presentation

The presentation slides are located in the `presentation/` folder:

📂 [`presentation/TechText_AI_Presentation.pptx`](presentation/TechText_AI_Presentation.pptx)

---

## Author

**Hussein** — Build Fellowship, 2025–2026  
GitHub: [@hussein3bak](https://github.com/hussein3bak)
