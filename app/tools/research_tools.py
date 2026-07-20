from __future__ import annotations
import json
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from llama_index.core.tools import FunctionTool
from app.config import config
from app.services.index_service import load_query_engine


class ToolBudgetExceeded(RuntimeError):
    pass


class ToolCallBudget:
    def __init__(self, limit: int):
        self.limit = limit
        self.calls = 0
        self.exceeded = False

    def spend(self) -> None:
        if self.calls >= self.limit:
            self.exceeded = True
            raise ToolBudgetExceeded(f"Tool call budget of {self.limit} exceeded.")
        self.calls += 1


def save_report(report_id: str, title: str, content: str) -> str:
    # Saving is local and intentionally restricted to the reports directory.
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    safe_name = "".join(ch for ch in title.lower().replace(" ", "_") if ch.isalnum() or ch == "_")
    path = reports_dir / f"{safe_name[:60] or 'report'}_{report_id}.md"
    path.write_text(content, encoding="utf-8")
    return f"Report saved to {path}"
def record_audit_event(report_id: str, action: str, detail: str) -> str:
    # Every consequential action should leave an auditable trace.
    log_path = Path("reports/audit_log.jsonl")
    log_path.parent.mkdir(exist_ok=True)
    event = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "report_id": report_id,
    "action": action,
    "detail": detail,
    }
    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, ensure_ascii=False) + "\n")
        return "Audit event recorded."


def search_knowledge(budget, query_engine, query: str) -> str:
    budget.spend()
    response = query_engine.query(query)
    names = sorted({n.node.metadata.get("file_name", "unknown") for n in response.source_nodes})
    return f"{str(response).strip()}\n\nSources: {', '.join(names) or 'none'}"

def _summarize(query_engine, topic: str) -> dict:
    response = query_engine.query(topic)
    sources = [
        {"source": n.node.metadata.get("file_name", "unknown"), "score": n.score}
        for n in response.source_nodes
    ]
    return {"topic": topic, "answer": str(response).strip(), "sources": sources}


def compare_sources(budget, query_engine, topic_a: str, topic_b: str) -> str:
    # Comparison is read-only by design: it must never touch the filesystem.
    budget.spend()
    side_a = _summarize(query_engine, topic_a)
    side_b = _summarize(query_engine, topic_b)
    names_a = {source["source"] for source in side_a["sources"]}
    names_b = {source["source"] for source in side_b["sources"]}

    limitations = []
    for side in (side_a, side_b):
        if not side["sources"]:
            limitations.append(
                f"No indexed sources were retrieved for '{side['topic']}'; "
                "claims about it are unsupported by the knowledge base."
            )
        elif len(side["sources"]) == 1:
            limitations.append(
                f"Only a single source supports '{side['topic']}'; findings may not generalize."
            )
    if not names_a & names_b:
        limitations.append(
            "The topics have no shared sources, so the comparison rests on separate evidence pools."
        )

    comparison = {
        "topic_a": side_a,
        "topic_b": side_b,
        "overlap": {"shared_sources": sorted(names_a & names_b)},
        "differences": {
            "sources_only_for_topic_a": sorted(names_a - names_b),
            "sources_only_for_topic_b": sorted(names_b - names_a),
        },
        "evidence_limitations": limitations,
    }
    return json.dumps(comparison, ensure_ascii=False, indent=2)


def build_tools(report_id: str = "unassigned", approved_to_save: bool = False, query_engine=None, budget=None):
    if query_engine is None:
        query_engine = load_query_engine()
    if budget is None:
        budget = ToolCallBudget(config.max_tool_calls)
    knowledge_tool = FunctionTool.from_defaults(
    fn=partial(search_knowledge, budget, query_engine),
    name="knowledge_base_search",
    description=(
    "Search the indexed organization knowledge base. Use it before making factual claims "
    "and return source-grounded findings."
    ),
    )
    audit_tool = FunctionTool.from_defaults(
    fn=partial(record_audit_event, report_id),
    name="record_audit_event",
    description="Record a concise audit event for important agent actions.",
    )
    compare_tool = FunctionTool.from_defaults(
    fn=partial(compare_sources, budget, query_engine),
    name="compare_sources",
    description=(
    "Compare two topics against the knowledge base. Returns JSON with each topic's "
    "source-grounded answer, shared and topic-specific sources, and evidence limitations. "
    "Read-only: it never writes files."
    ),
    )
    tools = [knowledge_tool, audit_tool, compare_tool]
    if approved_to_save:
        tools.append(
            FunctionTool.from_defaults(
            fn=partial(save_report, report_id),
            name="save_report",
            description="Save an approved Markdown report to the local reports directory.",
            )
        )
    return tools