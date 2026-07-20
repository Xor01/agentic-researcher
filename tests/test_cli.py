import asyncio

from app import cli
from app.models import ResearchRequest, Status


def run_cli_with(monkeypatch, inputs, request):
    feed = iter(inputs)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(feed))

    async def fake_run_research(question):
        return request

    monkeypatch.setattr(cli, "run_research", fake_run_research)
    monkeypatch.setattr(cli, "approve", lambda req: f"SAVED {req.report_id}")
    asyncio.run(cli.main())


def test_cli_shows_draft_and_saves_on_approval(monkeypatch, capsys):
    request = ResearchRequest(
        objective="q", status=Status.AWAITING_APPROVAL, report="THE DRAFT"
    )
    run_cli_with(monkeypatch, ["compare data policies question", "y", "exit"], request)
    out = capsys.readouterr().out
    assert "THE DRAFT" in out
    assert f"SAVED {request.report_id}" in out


def test_cli_reports_failure_reason(monkeypatch, capsys):
    request = ResearchRequest(objective="q", status=Status.FAILED, failure_reason="too broad")
    run_cli_with(monkeypatch, ["some rejected question here", "exit"], request)
    out = capsys.readouterr().out
    assert "too broad" in out
    assert "THE DRAFT" not in out
