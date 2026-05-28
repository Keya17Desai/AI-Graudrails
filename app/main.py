from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from app.api.chat import router as chat_router
from app.api.auth import router as auth_router
from app.api.eval import router as eval_router
from app.monitoring.middleware import PrometheusMiddleware
from app.core.config import settings
from app.ui.gradio_app import build as build_ui
import gradio as gr

app = FastAPI(
    title=settings.app_name,
    description="Internal company chatbot with RBAC and RAG",
    version="0.1.0",
)

app.add_middleware(PrometheusMiddleware)

app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(eval_router)

# Mount Gradio UI at /ui — accessible at http://localhost:8000/ui
gradio_app = build_ui()
app = gr.mount_gradio_app(app, gradio_app, path="/ui")


@app.get("/health")
def health():
    return {"status": "ok", "env": settings.app_env}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
