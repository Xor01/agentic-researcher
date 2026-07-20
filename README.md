# EvidenceOps — Agentic Researcher

A research agent that answers questions strictly from an indexed corpus of local
documents, drafts a report, and **saves nothing without explicit human approval**.

Built with LlamaIndex (`FunctionAgent`) over an OpenAI LLM, with a FastAPI service
and an interactive CLI.

## How it works

```text
CLI / API  →  orchestrator  →  research agent  →  tools  →  services
                   │                                │
        validation, status,                knowledge_base_search
        approval gate, budget              compare_sources
                                           record_audit_event
                                           save_report  (approved runs only)
```

The orchestrator owns the workflow: it validates the objective, builds a
request-scoped tool set, runs the agent, and gates saving behind human approval.
Every request carries a `report_id` that correlates the saved report with its
audit-log entries.

### Request lifecycle

| Status | Meaning |
| --- | --- |
| `draft` | Request created, not yet run |
| `awaiting_approval` | Agent produced a draft; nothing written yet |
| `approved` | Human approved; report saved (terminal) |
| `failed` | Objective rejected, tool budget exceeded, or an error (terminal) |

Transitions are enforced in [`app/models.py`](app/models.py) — approval is
single-use and cannot be reached without passing through `awaiting_approval`.

## Setup

Requires Python ≥ 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
# 1. install dependencies
uv sync

# 2. add your API key
cp .env.example .env      # then edit .env and set OPENAI_API_KEY

# 3. put source documents in data/
#    (PDFs; the folder is git-ignored)

# 4. build the vector index  (writes to storage/)
uv run python -m app.ingest
```

> **The index must be built before anything else runs.** `app.ingest` requires
> `llama-index-readers-file` — without it, PDFs are read as raw bytes and the
> index fills with unusable binary noise.

## Commands

| Command | What it does |
| --- | --- |
| `uv run python -m app.ingest` | Build/rebuild the vector index from `data/` |
| `uv run python -m app.cli` | Interactive research session (type `exit` to quit) |
| `uv run uvicorn app.api.main:app --reload` | Start the HTTP API on `:8000` |
| `uv run python -m pytest tests/ --ignore=tests/smoke_test.py` | Run the unit suite (offline, no API calls) |
| `uv run python -m tests.orchestrator_smoke` | Live end-to-end check (**costs API calls**) |
| `uv run python -m tests.compare_sources_smoke` | Live check of `compare_sources` (**costs API calls**) |
| `uv run python -m eval.run_eval` | Run the evaluation suite (**costs API calls**) — see [EVALUATION.md](EVALUATION.md) |

### API

```bash
# submit an objective — runs the agent, returns a draft (~25s)
curl -s -X POST http://127.0.0.1:8000/research \
  -H "Content-Type: application/json" \
  -d '{"objective": "compare data classification levels with personal data protection requirements"}'
# → {"report_id":"eb41f21f9100","status":"awaiting_approval","report":"...","failure_reason":""}

# review the draft, then approve using that report_id — this saves the file
curl -X POST http://127.0.0.1:8000/research/eb41f21f9100/approve
# → {"status":"approved","message":"Report saved to reports/..._eb41f21f9100.md"}
```

Interactive docs at `/docs`. Health check at `/health`.

Rejected objectives return **422** with the reason; approving an unknown or
already-approved id returns **404**.

### Configuration

Edit the `config` object in [`app/config.py`](app/config.py):

| Setting | Default | Purpose |
| --- | --- | --- |
| `llm_model` | `gpt-5.4` | Chat model for the agent |
| `embedding_model` | `text-embedding-3-small` | Embeddings for indexing/retrieval |
| `top_k` | `5` | Chunks retrieved per query |
| `max_tool_calls` | `10` | Tool-call budget per request |
| `data_dir` / `storage_dir` | `./data` / `./storage` | Source documents / persisted index |

## Safety design

Three mechanisms enforce the guarantees structurally, rather than by asking the
model to behave:

**Capability gating.** `build_tools()` omits `save_report` entirely unless
`approved_to_save=True`. The drafting agent has no save tool in its schema, so
no prompt — including text injected via a retrieved document — can make it
write a file. Saving happens in `approve()`, deterministically, outside the LLM.

**Bounded execution.** Each request gets a `ToolCallBudget`. Query tools spend
from it and raise past the limit; the orchestrator additionally checks the
budget after the run, so a framework that swallows tool errors still can't
produce an unbounded loop.

**Auditability.** Every consequential action appends a JSONL event to
`reports/audit_log.jsonl` with the request's `report_id`. Because the id is
bound into the tool at construction time, the agent cannot forge or omit it.

```bash
grep <report_id> reports/audit_log.jsonl   # full trail for any saved report
```

## Limitations

- **Arabic documents are effectively unreachable.** English queries do not retrieve
  the Arabic-language document, and the system answers from a substituted English
  document without flagging it. See [EVALUATION.md](EVALUATION.md#failure-analysis).
- **Corpus-bound.** The agent answers only from `data/`. It has no web access
  and no knowledge of anything outside the index. Retrieval quality caps answer
  quality: if a topic isn't indexed, the answer will be thin regardless of how
  capable the model is.
- **PDF extraction is imperfect.** Scanned or image-based PDFs yield little or
  no text and are silently under-represented. There is no OCR step. Verify new
  documents extract readable text before trusting answers about them.
- **In-memory approvals.** Pending drafts live in a per-process dict
  ([`app/api/main.py`](app/api/main.py)). A restart discards them, and running
  multiple workers breaks approval — the second call may hit a worker that never
  saw the draft. Single worker only, and approve before restarting.
- **No authentication.** The API has no authN/authZ, rate limiting, or per-user
  isolation. Anyone who can reach the port can spend your API budget and write
  files. Do not expose it beyond localhost as-is.
- **Cost and latency.** Each research request is a multi-step agent loop —
  roughly 25 seconds and several OpenAI calls. The index loads once per process
  (~6s on first query, cached afterwards).
- **Naive objective validation.** Rejection is heuristic (length, word count,
  catch-all phrases). It filters obvious cases, not all poorly-scoped ones.
- **No index invalidation.** Adding documents to `data/` requires re-running
  `app.ingest` manually; the app will not notice new files on its own.

## Ethical considerations

**Human approval is required by design.** The agent drafts; a person decides
what gets written. This is enforced by capability gating, not by instruction —
but it only covers *saving*. A person still has to actually read the draft
before approving. An approval clicked without reading defeats the entire control.

**Grounding is not a guarantee of truth.** The agent is instructed to cite
retrieved sources and to distinguish evidence from inference, and
`compare_sources` reports explicit evidence limitations. None of this eliminates
hallucination or misreading. Treat every output as a draft for expert review,
never as authoritative advice — especially for the policy, legal, and compliance
material this is aimed at, where a confident wrong answer carries real cost.

**Know your documents' sensitivity.** Document text is sent to OpenAI for
embedding and inference. Do not index confidential, personal, or regulated data
without confirming that transfer is permitted under the applicable policy — a
point the corpus in this repository happens to make about cross-border data
transfers.

**Provenance over fluency.** Answers should be verifiable against a named
source. The `Sources:` line and the `report_id` audit trail exist so any claim
can be traced back to a document and any saved report back to the run that
produced it. Preserve that trail if you extend the system.

**Retrieval shapes the answer.** With `top_k=5`, the agent sees a small slice of
the corpus. Absence of evidence in a response means "not retrieved," which is
not the same as "not in the documents," and is certainly not "false."

## Project layout

```text
app/
  cli.py             interactive session
  config.py          settings
  ingest.py          build the vector index from data/
  models.py          ResearchRequest, Status, transition rules
  orchestrator.py    validation, workflow, approval gate
  agents/            agent definition + system prompt
  api/               FastAPI service
  services/          LLM config, cached index/query engine
  tools/             agent tools (search, compare, audit, save)
tests/               unit tests (offline) + *_smoke.py (live, cost money)
eval/                evaluation dataset + harness  →  EVALUATION.md
data/                source documents
storage/             persisted vector index  (git-ignored)
reports/             saved reports + audit_log.jsonl  (git-ignored)
```
