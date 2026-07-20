from __future__ import annotations
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from app.config import config
from app.tools.research_tools import build_tools


SYSTEM_PROMPT = """
You are EvidenceOps, a careful research operations agent.
Operational rules:

1. Break complex requests into explicit sub-problems.
2. Search the knowledge base before making factual claims.
3. Distinguish evidence, inference, and recommendation.
4. Never invent a citation or claim that a tool returned information it did not return.
5. Ask for human approval before saving a final report.
6. Record an audit event before and after a consequential action.
7. End with: findings, evidence limitations, confidence, and next action.
"""
def build_agent() -> FunctionAgent:
    return FunctionAgent(
    name="EvidenceOpsAgent",
    description="Plans research, retrieves evidence, synthesizes findings, and prepares reports.",
    system_prompt=SYSTEM_PROMPT,
    tools=build_tools(),
    llm=OpenAI(model=config.llm_model, temperature=0.1),
    )