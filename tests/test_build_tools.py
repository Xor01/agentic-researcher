from types import SimpleNamespace

from app.tools.research_tools import build_tools


fake_engine = SimpleNamespace(query=lambda text: None)


def tool_names(tools):
    return [tool.metadata.name for tool in tools]


def test_build_tools_excludes_save_tool_by_default():
    names = tool_names(build_tools(query_engine=fake_engine))
    assert "save_report" not in names
    assert "knowledge_base_search" in names
    assert "record_audit_event" in names
    assert "compare_sources" in names


def test_build_tools_includes_save_tool_when_approved():
    names = tool_names(build_tools(approved_to_save=True, query_engine=fake_engine))
    assert "save_report" in names
