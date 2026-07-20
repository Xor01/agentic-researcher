from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4


class Status(str, Enum):
    DRAFT = "draft"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    FAILED = "failed"


_ALLOWED_TRANSITIONS = {
    Status.DRAFT: {Status.AWAITING_APPROVAL, Status.FAILED},
    Status.AWAITING_APPROVAL: {Status.APPROVED, Status.FAILED},
    Status.APPROVED: set(),
    Status.FAILED: set(),
}


def _new_report_id() -> str:
    return uuid4().hex[:12]


@dataclass
class ResearchRequest:
    objective: str
    report_id: str = field(default_factory=_new_report_id)
    status: Status = Status.DRAFT
    report: str = ""
    failure_reason: str = ""


def transition(request: ResearchRequest, new_status: Status) -> None:
    if new_status not in _ALLOWED_TRANSITIONS[request.status]:
        raise ValueError(
            f"Illegal status transition: {request.status.value} -> {new_status.value}"
        )
    request.status = new_status
