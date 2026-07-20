"""Evaluation harness: retrieval accuracy, groundedness, latency, and token cost.

Usage:
    uv run python -m eval.run_eval            # full dataset + agent subset
    uv run python -m eval.run_eval --no-agent # skip the expensive agent runs

Writes eval/results.json. Costs real OpenAI calls.
"""
from __future__ import annotations
import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import tiktoken
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.llms.openai import OpenAI

from app.config import config
from app.orchestrator import run_research, validate_objective
from app.services.index_service import load_query_engine

DATASET_PATH = Path("eval/dataset.jsonl")
RESULTS_PATH = Path("eval/results.json")

# Set these to the rates for your model; token counts below are measured, dollars are derived.
PRICE_INPUT_PER_1M = 1.25
PRICE_OUTPUT_PER_1M = 10.00

AGENT_SUBSET = ["cls-01", "prv-02", "cmp-02", "dfk-01", "oos-01"]

JUDGE_PROMPT = """You are grading a retrieval-augmented answer.

QUESTION:
{question}

RETRIEVED CONTEXT (the only material the system was given):
{context}

ANSWER:
{answer}

Reply with ONLY a JSON object, no prose:
{{"answered": true|false, "grounded": true|false}}

- "answered": false if the answer declines, says the information is unavailable,
  unreadable, or not in the provided material. true if it asserts substantive content.
- "grounded": true if the answer's claims are supported by the retrieved context.
  false if it asserts facts absent from the context (or if answered is false).
"""


def load_dataset() -> list[dict]:
    lines = DATASET_PATH.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines]


def make_token_handler() -> TokenCountingHandler:
    handler = TokenCountingHandler(tokenizer=tiktoken.get_encoding("cl100k_base").encode)
    Settings.callback_manager = CallbackManager([handler])
    return handler


def snapshot(handler: TokenCountingHandler) -> tuple[int, int, int]:
    return (
        handler.prompt_llm_token_count,
        handler.completion_llm_token_count,
        handler.total_embedding_token_count,
    )


def delta(before: tuple[int, int, int], after: tuple[int, int, int]) -> dict:
    return {
        "prompt_tokens": after[0] - before[0],
        "completion_tokens": after[1] - before[1],
        "embedding_tokens": after[2] - before[2],
    }


def estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (
        prompt_tokens / 1_000_000 * PRICE_INPUT_PER_1M
        + completion_tokens / 1_000_000 * PRICE_OUTPUT_PER_1M
    )


def judge_answer(judge_llm, question: str, context: str, answer: str) -> dict:
    # Context must not be truncated below what the system actually saw, or the judge
    # reports false "ungrounded" verdicts for correct answers.
    prompt = JUDGE_PROMPT.format(question=question, context=context[:20000], answer=answer[:3000])
    raw = str(judge_llm.complete(prompt)).strip()
    start, end = raw.find("{"), raw.rfind("}")
    try:
        parsed = json.loads(raw[start : end + 1])
        return {"answered": bool(parsed["answered"]), "grounded": bool(parsed["grounded"])}
    except Exception:
        return {"answered": None, "grounded": None, "judge_raw": raw[:200]}


def run_retrieval_case(engine, judge_llm, handler, case: dict) -> dict:
    before = snapshot(handler)
    started = time.perf_counter()
    response = engine.query(case["question"])
    latency = time.perf_counter() - started
    usage = delta(before, snapshot(handler))

    sources = sorted({n.node.metadata.get("file_name", "unknown") for n in response.source_nodes})
    context = "\n\n".join(n.node.get_content() for n in response.source_nodes)
    answer = str(response).strip()

    judge_before = snapshot(handler)
    verdict = judge_answer(judge_llm, case["question"], context, answer)
    judge_usage = delta(judge_before, snapshot(handler))

    expected = case["expected_sources"]
    return {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "expected_sources": expected,
        "retrieved_sources": sources,
        "source_hit": bool(set(expected) & set(sources)) if expected else None,
        "answer": answer,
        "answered": verdict.get("answered"),
        "grounded": verdict.get("grounded"),
        "latency_s": round(latency, 2),
        "usage": usage,
        "judge_usage": judge_usage,
        "cost_usd": round(estimate_cost(usage["prompt_tokens"], usage["completion_tokens"]), 6),
    }


def run_validation_cases() -> list[dict]:
    # Free: no model calls, these are rejected before any agent is built.
    probes = [
        ("", True),
        ("   ", True),
        ("AI", True),
        ("tell me everything about the knowledge base", True),
        ("compare policies " * 60, True),
        ("What are the data classification levels used by the policy?", False),
    ]
    results = []
    for text, should_reject in probes:
        reason = validate_objective(text)
        results.append({
            "objective": text[:60],
            "should_reject": should_reject,
            "rejected": reason is not None,
            "correct": (reason is not None) == should_reject,
            "reason": reason,
        })
    return results


async def run_agent_case(handler, case: dict) -> dict:
    before = snapshot(handler)
    started = time.perf_counter()
    request = await run_research(case["question"])
    latency = time.perf_counter() - started
    usage = delta(before, snapshot(handler))
    return {
        "id": case["id"],
        "question": case["question"],
        "status": request.status.value,
        "latency_s": round(latency, 2),
        "usage": usage,
        "cost_usd": round(estimate_cost(usage["prompt_tokens"], usage["completion_tokens"]), 6),
        "report_chars": len(request.report),
    }


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(round(pct / 100 * (len(ordered) - 1))), len(ordered) - 1)
    return ordered[index]


def summarize(cases: list[dict], validation: list[dict], agent_runs: list[dict]) -> dict:
    grounded_cases = [c for c in cases if c["expected_sources"]]
    hits = [c for c in grounded_cases if c["source_hit"]]
    oos = [c for c in cases if c["category"] == "out_of_scope"]
    abstained = [c for c in oos if c["answered"] is False]
    in_scope_answered = [c for c in grounded_cases if c["answered"]]
    in_scope_grounded = [c for c in grounded_cases if c["grounded"]]
    latencies = [c["latency_s"] for c in cases]
    prompt_tokens = sum(c["usage"]["prompt_tokens"] for c in cases)
    completion_tokens = sum(c["usage"]["completion_tokens"] for c in cases)
    embedding_tokens = sum(c["usage"]["embedding_tokens"] for c in cases)
    judge_prompt = sum(c["judge_usage"]["prompt_tokens"] for c in cases)
    judge_completion = sum(c["judge_usage"]["completion_tokens"] for c in cases)

    return {
        "n_cases": len(cases),
        "retrieval": {
            "n_with_expected_sources": len(grounded_cases),
            "source_hit_rate": round(len(hits) / len(grounded_cases), 3) if grounded_cases else None,
            "misses": [c["id"] for c in grounded_cases if not c["source_hit"]],
        },
        "answers": {
            "in_scope_answer_rate": round(len(in_scope_answered) / len(grounded_cases), 3) if grounded_cases else None,
            "in_scope_grounded_rate": round(len(in_scope_grounded) / len(grounded_cases), 3) if grounded_cases else None,
            "ungrounded_ids": [c["id"] for c in grounded_cases if c["grounded"] is False],
            "unanswered_ids": [c["id"] for c in grounded_cases if c["answered"] is False],
        },
        "out_of_scope": {
            "n": len(oos),
            "correct_abstention_rate": round(len(abstained) / len(oos), 3) if oos else None,
            "leaked_ids": [c["id"] for c in oos if c["answered"]],
        },
        "validation": {
            "n": len(validation),
            "accuracy": round(sum(v["correct"] for v in validation) / len(validation), 3),
        },
        "latency_s": {
            "mean": round(statistics.mean(latencies), 2) if latencies else 0,
            "p50": round(percentile(latencies, 50), 2),
            "p95": round(percentile(latencies, 95), 2),
            "max": round(max(latencies), 2) if latencies else 0,
        },
        "tokens": {
            "system_prompt": prompt_tokens,
            "system_completion": completion_tokens,
            "embedding": embedding_tokens,
            "judge_prompt": judge_prompt,
            "judge_completion": judge_completion,
        },
        "cost_usd": {
            "system_total": round(estimate_cost(prompt_tokens, completion_tokens), 4),
            "per_query_mean": round(estimate_cost(prompt_tokens, completion_tokens) / len(cases), 5) if cases else 0,
            "judge_total": round(estimate_cost(judge_prompt, judge_completion), 4),
            "rates_used": {"input_per_1m": PRICE_INPUT_PER_1M, "output_per_1m": PRICE_OUTPUT_PER_1M},
        },
        "agent": {
            "n": len(agent_runs),
            "latency_s_mean": round(statistics.mean([a["latency_s"] for a in agent_runs]), 2) if agent_runs else None,
            "latency_s_max": max([a["latency_s"] for a in agent_runs]) if agent_runs else None,
            "cost_usd_mean": round(statistics.mean([a["cost_usd"] for a in agent_runs]), 5) if agent_runs else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-agent", action="store_true", help="skip end-to-end agent runs")
    parser.add_argument("--limit", type=int, default=0, help="run only the first N cases")
    args = parser.parse_args()

    dataset = load_dataset()
    if args.limit:
        dataset = dataset[: args.limit]

    handler = make_token_handler()
    engine = load_query_engine()
    judge_llm = OpenAI(model=config.llm_model, temperature=0)

    cases = []
    for index, case in enumerate(dataset, start=1):
        result = run_retrieval_case(engine, judge_llm, handler, case)
        cases.append(result)
        flag = "" if result["source_hit"] in (True, None) else "  <-- MISS"
        print(f"[{index}/{len(dataset)}] {case['id']:<8} {result['latency_s']:>5.1f}s{flag}")

    validation = run_validation_cases()

    agent_runs = []
    if not args.no_agent:
        subset = [c for c in dataset if c["id"] in AGENT_SUBSET]
        for index, case in enumerate(subset, start=1):
            run = asyncio.run(run_agent_case(handler, case))
            agent_runs.append(run)
            print(f"[agent {index}/{len(subset)}] {case['id']:<8} {run['latency_s']:>5.1f}s  {run['status']}")

    payload = {
        "config": {
            "llm_model": config.llm_model,
            "embedding_model": config.embedding_model,
            "top_k": config.top_k,
            "max_tool_calls": config.max_tool_calls,
        },
        "metrics": summarize(cases, validation, agent_runs),
        "cases": cases,
        "validation": validation,
        "agent_runs": agent_runs,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {RESULTS_PATH}")
    print(json.dumps(payload["metrics"], indent=2))


if __name__ == "__main__":
    main()
