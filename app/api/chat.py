from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from app.rag.retriever import query
from app.auth.dependencies import get_current_user
from app.guardrails.pipeline import check_input, check_output, GuardrailViolation

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    session_id: str
    chunks_used: int = 0
    user_role: str


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user),
):
    # Input guardrails — PII and content safety
    try:
        check_input(request.message)
    except GuardrailViolation as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=e.detail)

    role = current_user["role"]
    result = query(question=request.message, user_role=role)

    # Output guardrails — catch PII leaking from documents
    try:
        check_output(result["answer"])
    except GuardrailViolation as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=e.detail)

    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        chunks_used=result["chunks_used"],
        session_id=request.session_id,
        user_role=role,
    )
