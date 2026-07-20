from __future__ import annotations
from app.agents.research_agent import build_agent
from app.config import config
from app.models import ResearchRequest, Status, transition
from app.tools.research_tools import (
    ToolBudgetExceeded,
    ToolCallBudget,
    build_tools,
    record_audit_event,
    save_report,
)

MIN_OBJECTIVE_WORDS = 4
MAX_OBJECTIVE_CHARS = 500
_BROAD_PHRASES = ("everything about", "all topics", "tell me everything")


def validate_objective(objective: str) -> str | None:
    # Rejection happens before any agent is built, so it costs zero tokens.
    text = objective.strip()
    if not text:
        return "Objective is empty."
    if len(text) > MAX_OBJECTIVE_CHARS:
        return f"Objective exceeds {MAX_OBJECTIVE_CHARS} characters; narrow it to one focused question."
    if len(text.split()) < MIN_OBJECTIVE_WORDS:
        return "Objective is too broad; state a specific question of at least four words."
    lowered = text.lower()
    for phrase in _BROAD_PHRASES:
        if phrase in lowered:
            return f"Objective is overly broad (contains '{phrase}'); narrow the scope."
    return None

async def run_research(objective: str, agent_factory=build_agent, query_engine=None) -> ResearchRequest:
    request = ResearchRequest(objective=objective)
    reason = validate_objective(objective)
    if reason:
        request.failure_reason = reason
        transition(request, Status.FAILED)
        return request

    # The draft agent gets no save capability; saving happens only in approve().
    budget = ToolCallBudget(config.max_tool_calls)
    tools = build_tools(report_id=request.report_id, query_engine=query_engine, budget=budget)
    agent = agent_factory(tools)
    prompt = f"""
Research objective: {objective}

Execution constraint: You cannot save reports. Return a draft and request approval.
Use the available tools and produce an evidence-grounded response.
"""
    try:
        result = await agent.run(user_msg=prompt)
    except ToolBudgetExceeded as error:
        request.failure_reason = str(error)
        transition(request, Status.FAILED)
        return request
    if budget.exceeded:
        request.failure_reason = f"Tool call budget of {budget.limit} exceeded during research."
        transition(request, Status.FAILED)
        return request

    request.report = str(result)
    transition(request, Status.AWAITING_APPROVAL)
    record_audit_event(request.report_id, "draft_created", objective)
    return request


def approve(request: ResearchRequest) -> str:
    # Single-use and scoped to this request: the transition check rejects anything
    # not awaiting approval, and nothing about the approval is persisted.
    transition(request, Status.APPROVED)
    message = save_report(request.report_id, request.objective, request.report)
    record_audit_event(request.report_id, "report_approved_and_saved", message)
    return message