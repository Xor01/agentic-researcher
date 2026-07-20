import json
from types import SimpleNamespace

from app.tools.research_tools import compare_sources


class FakeResponse:
    def __init__(self, answer, source_nodes):
        self.answer = answer
        self.source_nodes = source_nodes

    def __str__(self):
        return self.answer


def fake_node(file_name, score):
    return SimpleNamespace(
        node=SimpleNamespace(metadata={"file_name": file_name}),
        score=score,
    )


class FakeQueryEngine:
    def __init__(self, responses):
        self.responses = responses
        self.queries = []

    def query(self, text):
        self.queries.append(text)
        return self.responses[text]


def test_compare_sources_partitions_overlap_and_differences():
    engine = FakeQueryEngine(
        {
            "agent safety": FakeResponse(
                "Agents need guardrails.",
                [fake_node("guardrails.md", 0.9), fake_node("audit.md", 0.8)],
            ),
            "agent logging": FakeResponse(
                "Log every consequential action.",
                [fake_node("audit.md", 0.85), fake_node("logging.md", 0.7)],
            ),
        }
    )
    result = json.loads(compare_sources(engine, "agent safety", "agent logging"))

    assert engine.queries == ["agent safety", "agent logging"]
    assert result["topic_a"]["topic"] == "agent safety"
    assert result["topic_a"]["answer"] == "Agents need guardrails."
    assert result["topic_b"]["answer"] == "Log every consequential action."
    assert result["overlap"]["shared_sources"] == ["audit.md"]
    assert result["differences"]["sources_only_for_topic_a"] == ["guardrails.md"]
    assert result["differences"]["sources_only_for_topic_b"] == ["logging.md"]


def test_compare_sources_reports_evidence_limitations_when_sources_missing():
    engine = FakeQueryEngine(
        {
            "known topic": FakeResponse("An answer.", [fake_node("notes.md", 0.9)]),
            "unknown topic": FakeResponse("Empty Response", []),
        }
    )
    result = json.loads(compare_sources(engine, "known topic", "unknown topic"))

    limitations = " ".join(result["evidence_limitations"])
    assert "unknown topic" in limitations
    assert "shared" in limitations.lower()
    assert result["topic_b"]["sources"] == []


def test_compare_sources_writes_no_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    engine = FakeQueryEngine(
        {
            "a": FakeResponse("A.", [fake_node("a.md", 0.9)]),
            "b": FakeResponse("B.", [fake_node("b.md", 0.9)]),
        }
    )
    compare_sources(engine, "a", "b")

    assert list(tmp_path.iterdir()) == []
