"""
Run evaluation: python evaluate.py

Options:
  --quick   Skip RAGAS scoring (just run questions and show answers + latency)
  --cases N Run only the first N test cases
"""
import sys
import os

# Allow running from the project root without installing the package
sys.path.insert(0, os.path.dirname(__file__))

from app.evaluation.test_cases import EVAL_CASES
from app.evaluation.runner import run_all
from app.evaluation.report import save_and_print

quick = "--quick" in sys.argv
n_cases = None
for arg in sys.argv:
    if arg.startswith("--cases="):
        n_cases = int(arg.split("=")[1])

cases = EVAL_CASES[:n_cases] if n_cases else EVAL_CASES

print(f"Running {len(cases)} evaluation cases{'(quick mode — skipping RAGAS)' if quick else ''}...")
print()

runner_results = run_all(cases)

if quick:
    ragas_scores = {}
    print("\nSkipping RAGAS scoring (--quick mode).")
else:
    from app.evaluation.ragas_eval import run_ragas
    ragas_scores = run_ragas(runner_results)

save_and_print(runner_results, ragas_scores)
