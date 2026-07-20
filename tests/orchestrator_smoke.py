"""Live end-to-end smoke: run_research -> approve with the real agent and index.

Costs real OpenAI calls; run manually:
    uv run python -m tests.orchestrator_smoke

Named so pytest does not collect it (no test_ prefix).
"""
import asyncio

from app.models import Status
from app.orchestrator import approve, run_research

request = asyncio.run(
    run_research("Compare data classification levels with personal data protection requirements")
)
print("status:", request.status.value)
print("report_id:", request.report_id)
print()
print((request.report or request.failure_reason)[:1500])
print()
if request.status == Status.AWAITING_APPROVAL:
    print(approve(request))
