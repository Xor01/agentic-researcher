"""Live smoke test for compare_sources against the real index.

Costs real OpenAI calls; run manually:
    uv run python tests/compare_sources_smoke.py [topic_a] [topic_b]

Named so pytest does not collect it (no test_ prefix).
"""
import sys

from app.services.index_service import load_query_engine
from app.tools.research_tools import ToolCallBudget, compare_sources


topic_a = sys.argv[1] if len(sys.argv) > 2 else "data classification levels"
topic_b = sys.argv[2] if len(sys.argv) > 2 else "personal data protection requirements"

engine = load_query_engine()
print(compare_sources(ToolCallBudget(10), engine, topic_a, topic_b))
