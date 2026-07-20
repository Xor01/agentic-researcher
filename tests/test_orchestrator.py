import asyncio
import json
from types import SimpleNamespace

import pytest

from app.models import ResearchRequest, Status
from app.orchestrator import approve, run_research, validate_objective
from app.tools.research_tools import ToolBudgetExceeded


class FakeResponse:
    def __init__(self, answer):
        self.answer = answer
        self.source_nodes = []

    def __str__(self):
        return self.answer


fake_engine = SimpleNamespace(query=lambda text: FakeResponse("an answer"))

FOCUSED_OBJECTIVE = "compare data classification with personal data protection requirements"


class FakeAgent:
    def __init__(self, tools):
        self.tools = tools

    async def run(self, user_msg):
        return "# Draft report\ngrounded findings"


class RunawayAgent(FakeAgent):
    # Simulates a framework that feeds tool errors back to the LLM instead of raising.
    async def run(self, user_msg):
        knowledge = next(t for t in self.tools if t.metadata.name == "knowledge_base_search")
        for _ in range(15):
            try:
                knowledge.call(query="x")
            except ToolBudgetExceeded:
                pass
        return "runaway draft"


def test_rejected_objective_fails_without_building_agent():
    def exploding_factory(tools):
        raise AssertionError("agent must not be built for rejected objectives")

    request = asyncio.run(run_research("", agent_factory=exploding_factory))
    assert request.status == Status.FAILED
    assert request.failure_reason


def test_happy_path_awaits_approval_and_audits(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    request = asyncio.run(
        run_research(FOCUSED_OBJECTIVE, agent_factory=FakeAgent, query_engine=fake_engine)
    )
    assert request.status == Status.AWAITING_APPROVAL
    assert "Draft report" in request.report
    events = (tmp_path / "reports" / "audit_log.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(events[-1])["report_id"] == request.report_id


def test_draft_agent_has_no_save_capability():
    captured = {}

    def capturing_factory(tools):
        captured["names"] = [t.metadata.name for t in tools]
        return FakeAgent(tools)

    asyncio.run(run_research(FOCUSED_OBJECTIVE, agent_factory=capturing_factory, query_engine=fake_engine))
    assert "save_report" not in captured["names"]


def test_budget_exhaustion_fails_request_even_if_agent_swallows_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    request = asyncio.run(
        run_research(FOCUSED_OBJECTIVE, agent_factory=RunawayAgent, query_engine=fake_engine)
    )
    assert request.status == Status.FAILED
    assert "budget" in request.failure_reason.lower()


def test_approve_saves_correlated_report_once(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    request = ResearchRequest(
        objective=FOCUSED_OBJECTIVE, status=Status.AWAITING_APPROVAL, report="approved content"
    )
    approve(request)
    assert request.status == Status.APPROVED
    saved = list((tmp_path / "reports").glob("*.md"))
    assert len(saved) == 1
    assert request.report_id in saved[0].name
    assert saved[0].read_text(encoding="utf-8") == "approved content"
    events = (tmp_path / "reports" / "audit_log.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(events[-1])["report_id"] == request.report_id
    with pytest.raises(ValueError):
        approve(request)


def test_approve_requires_awaiting_approval_status():
    request = ResearchRequest(objective=FOCUSED_OBJECTIVE)
    with pytest.raises(ValueError):
        approve(request)


@pytest.mark.parametrize("objective", ["", "   ", "\n\t"])
def test_rejects_empty_objectives(objective):
    assert validate_objective(objective) is not None


@pytest.mark.parametrize("objective", ["AI", "data policy", "summarize documents"])
def test_rejects_too_short_objectives(objective):
    reason = validate_objective(objective)
    assert reason is not None and "broad" in reason.lower()


@pytest.mark.parametrize(
    "objective",
    [
        "tell me everything about the knowledge base",
        "summarize all topics in the documents",
    ],
)
def test_rejects_catch_all_objectives(objective):
    assert validate_objective(objective) is not None


def test_rejects_overlong_objectives():
    assert validate_objective("compare policies " * 60) is not None


def test_accepts_focused_objective():
    objective = "compare data classification levels with personal data protection requirements"
    assert validate_objective(objective) is None
