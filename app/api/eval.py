"""
POST /eval/run  — trigger an evaluation run from the API.
Requires c_level role (admin only).
Returns scores + per-question summary immediately (no background task).
"""
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from app.auth.dependencies import get_current_user
from app.evaluation.test_cases import EVAL_CASES
from app.evaluation.runner import run_all
from app.evaluation.report import save_and_print

router = APIRouter(prefix="/eval", tags=["evaluation"])


class EvalRequest(BaseModel):
    quick: bool = True        # skip RAGAS by default to keep the API fast
    max_cases: int | None = None


class EvalResponse(BaseModel):
    num_cases: int
    avg_latency_seconds: float
    ragas_scores: dict
    report_path: str


@router.post("/run", response_model=EvalResponse)
def run_eval(request: EvalRequest, current_user: dict = Depends(get_current_user)):
    if current_user["role"] != "c_level":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Evaluation runs are restricted to c_level users.",
        )

    cases = EVAL_CASES[:request.max_cases] if request.max_cases else EVAL_CASES
    runner_results = run_all(cases)

    if request.quick:
        ragas_scores = {}
    else:
        from app.evaluation.ragas_eval import run_ragas
        ragas_scores = run_ragas(runner_results)

    report_path = save_and_print(runner_results, ragas_scores)
    avg_latency = round(sum(r["latency_seconds"] for r in runner_results) / len(runner_results), 2)

    return EvalResponse(
        num_cases=len(runner_results),
        avg_latency_seconds=avg_latency,
        ragas_scores=ragas_scores,
        report_path=report_path,
    )
