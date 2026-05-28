"""
PII detection and redaction using Microsoft Presidio.
Detects entities like names, emails, phone numbers, credit cards, SSNs, etc.
"""
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import RecognizerResult, OperatorConfig

# Initialise once at module load — these are expensive to create
_analyzer = AnalyzerEngine()
_anonymizer = AnonymizerEngine()

# PII types we care about in a corporate chatbot context
ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "US_SSN",
    "IP_ADDRESS",
    "LOCATION",
    "DATE_TIME",
    "NRP",          # Nationality, religion, political group
    "IBAN_CODE",
    "MEDICAL_LICENSE",
]


def detect_pii(text: str) -> list[dict]:
    """Return a list of detected PII entities with type, score, and position."""
    results = _analyzer.analyze(text=text, language="en", entities=ENTITIES)
    return [
        {
            "entity_type": r.entity_type,
            "score": round(r.score, 2),
            "start": r.start,
            "end": r.end,
            "value": text[r.start:r.end],
        }
        for r in results
    ]


def redact_pii(text: str) -> str:
    """Replace PII in text with <ENTITY_TYPE> placeholders."""
    results = _analyzer.analyze(text=text, language="en", entities=ENTITIES)
    if not results:
        return text
    # Presidio's default replace operator uses <entity_type> as the placeholder
    operators = {
        entity: OperatorConfig("replace", {"new_value": f"<{entity}>"})
        for entity in ENTITIES
    }
    anonymized = _anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators=operators,
    )
    return anonymized.text


def has_pii(text: str, min_score: float = 0.7) -> bool:
    """Return True if any PII is detected above the confidence threshold."""
    results = _analyzer.analyze(text=text, language="en", entities=ENTITIES)
    return any(r.score >= min_score for r in results)
