from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.models import ResearchRequest, Status
from app.orchestrator import approve, run_research

app = FastAPI(title="EvidenceOps Agent API", version="1.0.0")

# Drafts awaiting approval, keyed by report_id. In-memory and per-process:
# approval is scoped to one request and does not survive a restart.
_pending: dict[str, ResearchRequest] = {}


class ObjectiveRequest(BaseModel):
    objective: str


class DraftResponse(BaseModel):
    report_id: str
    status: str
    report: str
    failure_reason: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/research", response_model=DraftResponse)
async def research(body: ObjectiveRequest) -> DraftResponse:
    request = await run_research(body.objective)
    if request.status == Status.FAILED:
        raise HTTPException(status_code=422, detail=request.failure_reason)
    _pending[request.report_id] = request
    return DraftResponse(
        report_id=request.report_id,
        status=request.status.value,
        report=request.report,
        failure_reason=request.failure_reason,
    )


@app.post("/research/{report_id}/approve")
def approve_report(report_id: str) -> dict[str, str]:

    request = _pending.pop(report_id, None)
    if request is None:
        raise HTTPException(status_code=404, detail="No report awaiting approval for that id.")
    message = approve(request)
    return {"status": request.status.value, "message": message}
