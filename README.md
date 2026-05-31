---
title: AI Graudrails
emoji: 💬
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

<div align="center">

# AI Graudrails

### An internal company chatbot built with RAG, RBAC, guardrails, monitoring, and evaluation — entirely on free, local tools.

[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-AIGraudrails-FFD21E?style=for-the-badge)](https://KeyaDesai17-AIGraudrails.hf.space/ui)
[![HuggingFace Space](https://img.shields.io/badge/HuggingFace-Space-orange?style=for-the-badge&logo=huggingface)](https://huggingface.co/spaces/KeyaDesai17/AIGraudrails)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?style=for-the-badge&logo=github)](https://github.com/Keya17Desai/AI-Graudrails)

</div>

---

## What is this?

AI Graudrails is a production-grade internal chatbot for a fictional company. Employees ask questions in natural language and get answers sourced directly from company documents. Each user only sees documents their role permits. Every request is screened for PII and unsafe content before it reaches the AI.

This project was built as a complete learning exercise covering the full stack of a real enterprise AI system — from document ingestion to deployment.

---

## How it works

1. **You ask a question** via the chat UI
2. The question is **screened for PII and unsafe content** — blocked if flagged
3. The system **searches company documents** using semantic similarity, filtered by your role
4. The retrieved document chunks are **handed to an LLM** which generates an answer
5. The answer is **screened for PII** before being returned to you
6. Every step is **logged to Prometheus** and visible in Grafana

---

## What's built inside

| Layer | What it does |
|-------|-------------|
| **Chat UI** | Gradio interface — login, chat, see document sources |
| **RAG Pipeline** | ChromaDB vector search + sentence-transformers embeddings |
| **RBAC** | 5 roles, enforced at the database retrieval layer |
| **JWT Auth** | Secure login with bcrypt passwords and signed tokens |
| **PII Guardrails** | Microsoft Presidio + spaCy detects and blocks personal data |
| **Safety Guardrails** | LLM-as-judge blocks prompt injection and jailbreak attempts |
| **Monitoring** | Prometheus + Grafana, 10-panel dashboard, 7 custom metrics |
| **Evaluation** | RAGAS-style scoring: faithfulness, relevancy, precision, recall |
| **CI/CD** | GitHub Actions runs tests and deploys to HuggingFace on every push |

---

## Live Demo

| Description | URL |
|-------------|-----|
| Chat UI | https://KeyaDesai17-AIGraudrails.hf.space/ui |
| API Docs | https://KeyaDesai17-AIGraudrails.hf.space/docs |
| HuggingFace Space | https://huggingface.co/spaces/KeyaDesai17/AIGraudrails |

### Demo credentials

| Username | Password | Role | Can see |
|----------|----------|------|---------|
| alice | alice123 | employee | Company handbook, IT policy, code of conduct |
| bob | bob123 | hr | + Payroll and benefits |
| carol | carol123 | finance | + Budget and financial reports |
| dave | dave123 | marketing | + Marketing reports and expenses |
| eve | eve123 | c_level | Everything |

---

## Tech Stack

| | Tool |
|--|------|
| UI | Gradio |
| API | FastAPI |
| LLM | Ollama (local) / Groq free tier (cloud fallback) |
| Vector DB | ChromaDB |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` |
| Auth | JWT + bcrypt |
| PII Detection | Microsoft Presidio + spaCy `en_core_web_lg` |
| Safety | LLM-as-judge (llama3) |
| Monitoring | Prometheus + Grafana |
| Evaluation | Custom RAGAS-style metrics |
| CI/CD | GitHub Actions |
| Deployment | HuggingFace Spaces (Docker) |

---

## Run locally

```bash
# Clone
git clone https://github.com/Keya17Desai/AI-Graudrails.git
cd AI-Graudrails

# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_lg

# Configure
cp .env.example .env   # edit with your values

# Start ChromaDB
docker-compose up -d chromadb

# Ingest documents
python ingest.py data/documents

# Start server
uvicorn app.main:app --reload
```

Open **http://localhost:8000/ui**

---

## Document structure

```
data/documents/
  general/     →  all roles
  hr/          →  hr and c_level only
  finance/     →  finance and c_level only
  marketing/   →  marketing, finance, and c_level
```

---

*Built by [Keya Desai](https://github.com/Keya17Desai)*
