FROM python:3.10-slim

# HuggingFace Spaces requires a non-root user
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake spaCy model into the image at build time
# This means it's ready instantly on startup — no 400MB download on every wake-up
RUN python -m spacy download en_core_web_lg

# Copy application code and documents
COPY --chown=appuser:appuser . .

USER appuser

# HuggingFace Spaces requires port 7860
EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
