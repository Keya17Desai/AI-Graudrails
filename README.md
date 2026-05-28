---
title: Nexora Internal Chatbot
emoji: 💬
colorFrom: blue
colorTo: indigo
sdk: docker
dockerfile: Dockerfile.spaces
app_port: 7860
pinned: false
---

# Nexora Technologies — Internal Company Chatbot

An internal RAG chatbot with a chat UI, JWT authentication, role-based access control, PII guardrails, Prometheus monitoring, and RAGAS-style evaluation. Built entirely on free, locally-runnable tools.

## UI

Open `/ui` in your browser for a chat interface — no API client needed.

Login with any demo user, ask questions, and see answers sourced from company documents.
Guardrail blocks (PII in your message, unsafe content) are shown inline in the chat.

## Demo Users

| Username | Password  | Role      | Can access |
|----------|-----------|-----------|------------|
| alice    | alice123  | employee  | General docs (handbook, IT policy, code of conduct) |
| bob      | bob123    | hr        | General + HR (payroll, benefits) |
| carol    | carol123  | finance   | General + Finance (budget, Q1 report) |
| dave     | dave123   | marketing | General + Marketing + Finance |
| eve      | eve123    | c_level   | Everything |

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/ui` | Chat UI (Gradio) |
| `POST` | `/auth/login` | Get a JWT token |
| `GET`  | `/auth/me` | Check current user |
| `POST` | `/chat` | Send a message (requires Bearer token) |
| `POST` | `/eval/run` | Run evaluation (c_level only) |
| `GET`  | `/health` | Health check |
| `GET`  | `/metrics` | Prometheus metrics |
| `GET`  | `/docs` | Swagger API docs |

## Features

- **Chat UI** — Gradio interface at `/ui`, handles login and token management automatically
- **RAG** — ChromaDB vector search + sentence-transformers embeddings, answers from company documents
- **RBAC** — role-based access enforced at the retrieval layer (ChromaDB metadata filters), not the LLM layer
- **JWT Auth** — bcrypt password hashing, HS256 signed tokens, 8-hour expiry
- **Input Guardrails** — PII detection (Presidio + spaCy) blocks personal data in questions; LLM-as-judge blocks prompt injection
- **Output Guardrails** — PII scan on every LLM response before it reaches the caller
- **Monitoring** — Prometheus counters and histograms for requests, latency, guardrail blocks, LLM speed, RAG quality
- **Evals** — faithfulness, answer relevancy, context precision, context recall (llama3 as judge)

## Local Development

```bash
# 1. Start ChromaDB
docker-compose up -d chromadb

# 2. Ingest company documents into ChromaDB
python ingest.py data/documents

# 3. Start the server
uvicorn app.main:app --reload

# 4. Open the UI
open http://localhost:8000/ui

# Optional: start monitoring stack
docker-compose up -d prometheus grafana
# Grafana dashboard → http://localhost:3000/d/chatbot-main

# Optional: run evaluations
python evaluate.py --quick          # just answers + latency
python evaluate.py                  # full RAGAS scoring (slow, uses LLM judge)
```

## Deploying to HuggingFace Spaces

1. Create a new Space — type: **Docker**
2. Add these secrets in Space Settings → Variables and secrets:

| Secret | Value |
|--------|-------|
| `JWT_SECRET_KEY` | Any long random string |
| `GROQ_API_KEY` | Free key from [console.groq.com](https://console.groq.com) |
| `CHROMA_MODE` | `embedded` |

3. Push this repo to the Space:

```bash
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
git push hf main
```

HuggingFace builds the Docker image (~5 min). Your app is then live at `https://your-username-your-space-name.hf.space/ui`.

## Document Structure

```
data/documents/
  general/      → all roles (employee handbook, IT policy, code of conduct)
  hr/           → hr + c_level only
  finance/      → finance + c_level only
  marketing/    → marketing + finance + c_level
```

Add new documents to the appropriate folder and re-run `python ingest.py data/documents`.

## Stack

| Layer | Tool |
|-------|------|
| UI | Gradio |
| API | FastAPI |
| LLM | Ollama (local) / Groq (cloud fallback) |
| Vector DB | ChromaDB |
| Embeddings | sentence-transformers all-MiniLM-L6-v2 |
| Auth | JWT (python-jose) + bcrypt (passlib) |
| PII Detection | Microsoft Presidio + spaCy en_core_web_lg |
| Safety | llama3 as LLM-as-judge |
| Monitoring | Prometheus + Grafana |
| Evaluation | Custom RAGAS-style metrics (faithfulness, relevancy, precision, recall) |
| CI | GitHub Actions |
| Deploy | HuggingFace Spaces (Docker) |
