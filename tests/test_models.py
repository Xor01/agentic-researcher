import pytest

from app.models import ResearchRequest, Status, transition


def test_new_request_starts_as_draft_with_unique_id():
    first = ResearchRequest(objective="compare data policies")
    second = ResearchRequest(objective="compare data policies")
    assert first.status == Status.DRAFT
    assert first.report_id and first.report_id != second.report_id


def test_legal_transitions():
    request = ResearchRequest(objective="x")
    transition(request, Status.AWAITING_APPROVAL)
    assert request.status == Status.AWAITING_APPROVAL
    transition(request, Status.APPROVED)
    assert request.status == Status.APPROVED


def test_any_active_state_can_fail():
    for start in (Status.DRAFT, Status.AWAITING_APPROVAL):
        request = ResearchRequest(objective="x", status=start)
        transition(request, Status.FAILED)
        assert request.status == Status.FAILED


@pytest.mark.parametrize(
    ("start", "target"),
    [
        (Status.DRAFT, Status.APPROVED),          # cannot skip approval gate
        (Status.APPROVED, Status.APPROVED),       # approval is single-use
        (Status.APPROVED, Status.AWAITING_APPROVAL),
        (Status.FAILED, Status.AWAITING_APPROVAL),
    ],
)
def test_illegal_transitions_raise(start, target):
    request = ResearchRequest(objective="x", status=start)
    with pytest.raises(ValueError):
        transition(request, target)
    assert request.status == start
