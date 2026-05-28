---
title: Nexora Internal Chatbot
emoji: 💬
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

<div align="center">

# Nexora Internal Chatbot

**A production-grade internal RAG chatbot with RBAC, guardrails, monitoring, and evaluation.**

Built entirely on free, locally-runnable tools — no OpenAI required.

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![CI](https://img.shields.io/badge/CI-GitHub_Actions-2088FF?logo=github-actions)](https://github.com/features/actions)
[![Live Demo](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-FFD21E)](https://huggingface.co/spaces/KeyaDesai17/AIGraudrails)

</div>

---

## Overview

Nexora Internal Chatbot lets employees query company documents using natural language. It retrieves relevant information using semantic search, generates grounded answers using a local LLM, and enforces role-based access control so each user only sees documents they are authorised to read.

Every request passes through four independent layers of protection: JWT authentication, PII detection, content safety classification, and RBAC-filtered vector retrieval. The system is fully observable via Prometheus metrics and a pre-built Grafana dashboard.

---

## Features

| | Feature | Details |
|---|---|---|
| 💬 | **Chat UI** | Gradio interface — login, chat, see sources, all in the browser |
| 🔍 | **RAG** | ChromaDB + sentence-transformers, semantic search over company documents |
| 🔐 | **JWT Auth** | bcrypt passwords, HS256 signed tokens, 8-hour expiry |
| 👥 | **RBAC** | 5 roles, enforced at the ChromaDB retrieval layer — not the LLM layer |
| 🛡️ | **PII Guardrails** | Presidio + spaCy blocks personal data in inputs and outputs |
| 🚨 | **Safety Guardrails** | LLM-as-judge blocks prompt injection and jailbreak attempts |
| 📊 | **Monitoring** | Prometheus + Grafana, 7 metrics, 10-panel dashboard |
| 🧪 | **Evaluation** | RAGAS-style metrics: faithfulness, relevancy, precision, recall |
| 🚀 | **CI/CD** | GitHub Actions lint + test + Docker build + HuggingFace deploy |

---

## Quick Start

### Prerequisites

- Python 3.10+
- Docker + Docker Compose
- [Ollama](https://ollama.com) with `llama3` pulled (`ollama pull llama3`)

### Run locally

```bash
# 1. Clone the repo
git clone https://github.com/Keya17Desai/AI-Graudrails.git
cd AI-Graudrails

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_lg

# 4. Copy environment config
cp .env.example .env            # then edit .env with your values

# 5. Start infrastructure
docker-compose up -d chromadb

# 6. Ingest company documents
python ingest.py data/documents

# 7. Start the server
uvicorn app.main:app --reload
```

Open **http://localhost:8000/ui** for the chat interface.  
Open **http://localhost:8000/docs** for the Swagger API docs.

---

## Environment Variables

Create a `.env` file at the project root (copy from `.env.example`):

```env
APP_NAME=company-chatbot
APP_ENV=development

JWT_SECRET_KEY=your-long-random-secret-here
JWT_EXPIRE_MINUTES=480

CHROMA_HOST=localhost
CHROMA_PORT=8001
CHROMA_MODE=http                 # use "embedded" for HuggingFace Spaces

OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3

GROQ_API_KEY=                    # optional: Groq fallback when Ollama is unavailable
```

---

## Demo Users

| Username | Password | Role | Document Access |
|----------|----------|------|----------------|
| `alice` | `alice123` | employee | Handbook, IT policy, code of conduct |
| `bob` | `bob123` | hr | + Payroll & benefits |
| `carol` | `carol123` | finance | + Budget, Q1 financial report |
| `dave` | `dave123` | marketing | + Marketing reports & expenses |
| `eve` | `eve123` | c_level | Everything |

---

## API Reference

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/ui` | — | Gradio chat interface |
| `POST` | `/auth/login` | — | Login, returns JWT token |
| `GET` | `/auth/me` | Bearer | Current user info |
| `POST` | `/chat` | Bearer | Send a message, get a RAG answer |
| `POST` | `/eval/run` | Bearer (c_level) | Run evaluation suite |
| `GET` | `/health` | — | Health check |
| `GET` | `/metrics` | — | Prometheus metrics |
| `GET` | `/docs` | — | Swagger UI |

### Example

```bash
# 1. Login
TOKEN=$(curl -s -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "alice", "password": "alice123"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

# 2. Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"message": "What is the leave policy?"}'
```

---

## Project Structure

```
AI-Graudrails/
├── app/
│   ├── api/            # Route handlers (auth, chat, eval)
│   ├── auth/           # JWT, bcrypt, user store, dependencies
│   ├── core/           # Config (pydantic-settings, reads .env)
│   ├── evaluation/     # RAGAS-style eval metrics and runner
│   ├── guardrails/     # PII detection, safety classifier, pipeline
│   ├── llm/            # Ollama + Groq clients, auto-routing
│   ├── monitoring/     # Prometheus metrics and middleware
│   ├── rag/            # Embeddings, ChromaDB, ingestion, retriever
│   ├── ui/             # Gradio chat interface
│   └── main.py         # FastAPI app + Gradio mount
├── data/documents/     # Company documents, organised by role
│   ├── general/        # All roles
│   ├── hr/             # HR + c_level
│   ├── finance/        # Finance + c_level
│   └── marketing/      # Marketing + finance + c_level
├── docker/             # Prometheus + Grafana config and dashboards
├── .github/workflows/  # CI (lint, test, Docker build) + deploy to HF Spaces
├── ingest.py           # CLI: ingest documents into ChromaDB
├── evaluate.py         # CLI: run evaluation suite
├── docker-compose.yml  # Local dev: ChromaDB, Postgres, Prometheus, Grafana
├── Dockerfile          # Production image (port 8000)
├── Dockerfile.spaces   # HuggingFace Spaces image (port 7860, embedded ChromaDB)
└── requirements.txt
```

---

## Monitoring

Start the full monitoring stack:

```bash
docker-compose up -d prometheus grafana
```

| Service | URL |
|---------|-----|
| Prometheus | http://localhost:9090 |
| Grafana dashboard | http://localhost:3000/d/chatbot-main |

The Grafana dashboard auto-provisions on startup with 10 panels covering request rate, latency percentiles, guardrail block rate, LLM response time, RAG chunk retrieval, and auth failure rate.

---

## Evaluation

```bash
# Quick run — answers + latency only, no LLM judge
python evaluate.py --quick

# Full RAGAS-style scoring (uses llama3 as judge, takes ~10 min)
python evaluate.py

# Limit to first N test cases
python evaluate.py --cases=4
```

Results are saved to `results/eval_<timestamp>.json`.

| Metric | What it measures |
|--------|-----------------|
| Faithfulness | Are all answer claims supported by retrieved chunks? |
| Answer Relevancy | Is the answer on-topic for the question? |
| Context Precision | Are the retrieved chunks relevant to the question? |
| Context Recall | Do the chunks cover the full ground truth answer? |

---

## Deploying to HuggingFace Spaces

### 1. Create a Space
Go to [huggingface.co/new-space](https://huggingface.co/new-space) → SDK: **Docker** → Visibility: Private.

### 2. Add Space secrets
In Space Settings → Variables and secrets:

| Secret | Value |
|--------|-------|
| `JWT_SECRET_KEY` | Any long random string |
| `GROQ_API_KEY` | Free key from [console.groq.com](https://console.groq.com) |
| `CHROMA_MODE` | `embedded` |
| `OLLAMA_MODEL` | `llama3` |

### 3. Add GitHub Actions secrets
In GitHub repo → Settings → Secrets → Actions:

| Secret | Value |
|--------|-------|
| `HF_TOKEN` | HuggingFace write token from [hf.co/settings/tokens](https://huggingface.co/settings/tokens) |
| `HF_USERNAME` | Your HuggingFace username |
| `HF_SPACE_NAME` | Your Space name (exactly as created) |

### 4. Deploy
Push to `main` — GitHub Actions runs CI then automatically pushes to HF Spaces:

```bash
git push origin main
```

Monitor build progress on your Space page. Once status shows **Running**, the app is live at:
```
https://KeyaDesai17-AIGraudrails.hf.space/ui
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Gradio 6 |
| API | FastAPI |
| LLM | Ollama (local) / Groq free tier (cloud fallback) |
| Vector DB | ChromaDB |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` |
| Auth | python-jose (JWT) + passlib (bcrypt) |
| PII Detection | Microsoft Presidio + spaCy `en_core_web_lg` |
| Safety | LLM-as-judge (llama3) |
| Monitoring | Prometheus + Grafana |
| Evaluation | Custom RAGAS-style (faithfulness, relevancy, precision, recall) |
| CI/CD | GitHub Actions |
| Deployment | HuggingFace Spaces (Docker) |

---

## License

MIT License — see [LICENSE](LICENSE) for details.
