"""
Starlette middleware that records http_requests_total and
http_request_duration_seconds for every request automatically.
"""
import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from app.monitoring.metrics import http_requests_total, http_request_duration_seconds

# Normalise dynamic path segments so /chat doesn't produce unbounded label cardinality
_KNOWN_PATHS = {"/chat", "/auth/login", "/auth/me", "/health", "/metrics"}


def _normalise(path: str) -> str:
    return path if path in _KNOWN_PATHS else "other"


class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        endpoint = _normalise(request.url.path)
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start

        http_requests_total.labels(
            method=request.method,
            endpoint=endpoint,
            status_code=str(response.status_code),
        ).inc()

        http_request_duration_seconds.labels(endpoint=endpoint).observe(duration)

        return response
