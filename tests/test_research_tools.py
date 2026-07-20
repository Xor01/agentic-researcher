import json
from types import SimpleNamespace

import pytest

from app.tools.research_tools import (
    ToolBudgetExceeded,
    ToolCallBudget,
    build_tools,
    record_audit_event,
    save_report,
)


class FakeResponse:
    def __init__(self, answer):
        self.answer = answer
        self.source_nodes = []

    def __str__(self):
        return self.answer


fake_engine = SimpleNamespace(query=lambda text: FakeResponse("an answer"))


def test_budget_allows_limit_then_raises_and_flags():
    budget = ToolCallBudget(2)
    budget.spend()
    budget.spend()
    assert budget.exceeded is False
    with pytest.raises(ToolBudgetExceeded):
        budget.spend()
    assert budget.exceeded is True


def test_knowledge_tool_consumes_budget(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    budget = ToolCallBudget(1)
    tools = build_tools(report_id="abc123", query_engine=fake_engine, budget=budget)
    knowledge = next(t for t in tools if t.metadata.name == "knowledge_base_search")
    knowledge.call(query="data classification")
    with pytest.raises(ToolBudgetExceeded):
        knowledge.call(query="data classification")


def test_save_report_filename_includes_report_id(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    message = save_report("abc123", "My Report", "content")
    saved = list((tmp_path / "reports").glob("*.md"))
    assert len(saved) == 1
    assert "abc123" in saved[0].name
    assert "abc123" in message


def test_audit_event_includes_report_id(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    record_audit_event("abc123", "draft_created", "some detail")
    lines = (tmp_path / "reports" / "audit_log.jsonl").read_text(encoding="utf-8").splitlines()
    event = json.loads(lines[-1])
    assert event["report_id"] == "abc123"
    assert event["action"] == "draft_created"
