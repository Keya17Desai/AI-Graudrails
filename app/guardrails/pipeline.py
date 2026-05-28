"""
Guardrail pipeline — runs all checks in sequence.
Call check_input() before RAG, check_output() before returning to caller.
"""
from app.guardrails.pii import has_pii, detect_pii
from app.guardrails.safety import check_safety
from app.monitoring.metrics import guardrail_blocks_total


class GuardrailViolation(Exception):
    """Raised when a guardrail blocks the request."""
    def __init__(self, reason: str, detail: str = ""):
        self.reason = reason
        self.detail = detail
        super().__init__(reason)


def check_input(text: str) -> None:
    """
    Run all input guardrails. Raises GuardrailViolation if any check fails.
    Runs PII check first (cheap), then safety check (LLM call).
    """
    if has_pii(text):
        entities = detect_pii(text)
        types = list({e["entity_type"] for e in entities})
        guardrail_blocks_total.labels(stage="input", reason="pii_in_input").inc()
        raise GuardrailViolation(
            reason="pii_in_input",
            detail=f"Your message contains personal information ({', '.join(types)}). "
                   "Please rephrase without personal data.",
        )

    result = check_safety(text)
    if not result["safe"]:
        guardrail_blocks_total.labels(stage="input", reason="unsafe_content").inc()
        raise GuardrailViolation(
            reason="unsafe_content",
            detail=f"Your message was flagged: {result['reason']}",
        )


def check_output(text: str) -> str:
    """
    Scan LLM output for PII before returning it to the caller.
    Returns the original text if clean; raises GuardrailViolation if PII found.
    """
    if has_pii(text):
        entities = detect_pii(text)
        types = list({e["entity_type"] for e in entities})
        guardrail_blocks_total.labels(stage="output", reason="pii_in_output").inc()
        raise GuardrailViolation(
            reason="pii_in_output",
            detail=f"Response was blocked because it contained personal information ({', '.join(types)}). "
                   "Please contact your administrator if this is unexpected.",
        )
    return text
