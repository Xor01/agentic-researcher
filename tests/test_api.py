import pytest
from fastapi.testclient import TestClient

from app.api import main
from app.models import ResearchRequest, Status


@pytest.fixture
def client(monkeypatch):
    main._pending.clear()

    async def fake_run_research(objective):
        if objective == "bad":
            return ResearchRequest(objective=objective, status=Status.FAILED,
                                   failure_reason="Objective is too broad.")
        return ResearchRequest(objective=objective, status=Status.AWAITING_APPROVAL,
                               report="THE DRAFT")

    monkeypatch.setattr(main, "run_research", fake_run_research)
    monkeypatch.setattr(main, "approve", lambda req: f"Report saved for {req.report_id}")
    return TestClient(main.app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_research_returns_draft_and_report_id(client):
    body = client.post("/research", json={"objective": "compare data policies"}).json()
    assert body["status"] == "awaiting_approval"
    assert body["report"] == "THE DRAFT"
    assert body["report_id"]


def test_rejected_objective_returns_422(client):
    response = client.post("/research", json={"objective": "bad"})
    assert response.status_code == 422
    assert "too broad" in response.json()["detail"]


def test_approve_saves_pending_report(client):
    report_id = client.post("/research", json={"objective": "compare data policies"}).json()["report_id"]
    response = client.post(f"/research/{report_id}/approve")
    assert response.status_code == 200
    assert report_id in response.json()["message"]


def test_approve_unknown_report_id_returns_404(client):
    assert client.post("/research/nonexistent/approve").status_code == 404


def test_approve_is_single_use(client):
    report_id = client.post("/research", json={"objective": "compare data policies"}).json()["report_id"]
    client.post(f"/research/{report_id}/approve")
    assert client.post(f"/research/{report_id}/approve").status_code == 404
