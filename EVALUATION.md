# Evaluation Report

Evaluation of the EvidenceOps research agent against its indexed corpus.

- **Dataset:** [`eval/dataset.jsonl`](eval/dataset.jsonl) — 36 questions
- **Harness:** [`eval/run_eval.py`](eval/run_eval.py) — `uv run python -m eval.run_eval`
- **Raw results:** `eval/results.json`
- **Configuration:** `llm_model=gpt-5.4`, `embedding_model=text-embedding-3-small`,
  `top_k=5`, `max_tool_calls=10`
- **Chunking:** `SentenceSplitter(chunk_size=700, chunk_overlap=100)` → 447 chunks
- **Corpus:** 10 PDFs (Saudi data governance, AI, and privacy policy documents)

## Dataset composition

| Category | N | What it tests |
| --- | --- | --- |
| `factual` | 30 | Single-document retrieval and answer accuracy |
| `comparison` | 2 | Synthesis across two documents |
| `out_of_scope` | 3 | Correct abstention when the corpus has no answer |
| `arabic_probe` | 1 | Retrieval against the one Arabic-language document |
| **Total** | **36** | |

Each question declares `expected_sources` — the document(s) that should appear in
the retrieved set. A retrieval "hit" means at least one expected source was retrieved
at `top_k=5`. Six additional validation probes (empty / too-short / catch-all /
over-long objectives) run without any model calls.

## Metrics

| Metric | Result |
| --- | --- |
| **Retrieval hit rate** (33 questions with expected sources) | **97.0%** (32/33) |
| **Groundedness** (in-scope answers supported by retrieved context) | **93.9%** (31/33) |
| **In-scope answer rate** (answered rather than declined) | **97.0%** (32/33) |
| **Correct abstention on out-of-scope** | **100%** (3/3) |
| **Objective-validation accuracy** | **100%** (6/6) |
| Retrieval misses | `ara-01` |
| Ungrounded | `ara-01`, `edu-02` |

### Latency

Retrieval-level (single query, index already loaded):

| | Seconds |
| --- | --- |
| Mean | 2.87 |
| p50 | 2.66 |
| p95 | 4.42 |
| Max | 7.41 |

Comparison questions are slower than factual ones — expected, since `compare_sources`
issues two queries.

Full agent loop (5-question subset, end-to-end through `run_research`):

| | Seconds |
| --- | --- |
| Mean | 28.35 |
| Max | 47.28 |

Index load costs an additional **~6s once per process** (cached thereafter). Before
the index was rebuilt it was ~145s *per query* — see [README](README.md#limitations).

### Cost

Token counts are measured; dollar figures are derived at the rates below and are only
as accurate as those rates.

| Component | Prompt tokens | Completion tokens | Cost (USD) |
| --- | --- | --- | --- |
| System under test (36 queries) | 65,388 | 4,882 | $0.1306 |
| LLM judge (36 gradings) | 62,070 | 501 | $0.0826 |
| Agent runs (5 end-to-end) | — | — | $0.1401 |
| Embeddings (36 queries) | 466 | — | negligible |

- **Mean cost per retrieval query: $0.0036**
- **Mean cost per full agent request: $0.0280**
- Rates assumed: **$1.25 / 1M input, $10.00 / 1M output**. Set
  `PRICE_INPUT_PER_1M` / `PRICE_OUTPUT_PER_1M` in [`eval/run_eval.py`](eval/run_eval.py)
  to your actual rates; token counts are unaffected.

An agent request costs ~6× a bare retrieval query, because the agent issues multiple
tool calls and carries a growing conversation into each one.

## Failure analysis

### 1. Arabic document is effectively unreachable — and fails silently (`ara-01`)

The single retrieval miss is the only Arabic-language document,
`organizationalArrangements.pdf`. The English query "What organizational arrangements
are described for data governance responsibilities?" retrieved
`DataManagementPersonalDataProtectionStandards.pdf` instead.

**The serious part is not the miss — it is that the system answered anyway.** It
produced a confident, well-formed answer about Data Management Offices and committees
drawn from the *substituted* document, with no indication that the document the
question was actually about had contributed nothing. A user cannot distinguish this
from a correct answer without checking the sources themselves.

**Cause:** `text-embedding-3-small` places Arabic and English text in poorly aligned
regions of the embedding space, so cross-language retrieval fails; the retriever then
returns the best *available* English match, and `top_k=5` always returns 5 chunks
regardless of how weak they are.

**Mitigations:** apply a similarity-score floor so weak matches are dropped rather than
passed off as evidence; translate Arabic documents at ingestion time or store bilingual
chunks; surface retrieval scores in the answer so thin evidence is visible.

### 2. Language drift on out-of-scope questions (`oos-03`)

Asked about Aramco's 2024 quarterly revenue (correctly, not in the corpus), the system
abstained — **in Arabic**, despite an English question. The abstention was counted as
correct, and semantically it was, but a user who does not read Arabic receives an
unusable response.

**Cause:** the retrieved chunks came from Arabic content, and the model followed the
language of its context rather than the language of the question.

**Mitigation:** pin the response language in the query prompt template.

### 3. Retrieval is strong where documents are unambiguous

All 30 English factual questions retrieved an expected source, including cases where
several documents cover adjacent ground (`GenerativeAIPublicEN.pdf` vs
`GenAIGuidelinesForGovernmentENCompressed.pdf`). Both comparison questions retrieved
both expected documents. This suggests `top_k=5` is adequate for this corpus size.

### 4. Honest partial answer counted as a failure (`edu-02`)

Asked which international benchmarks informed the Saudi academic framework, the system
retrieved the correct document (`File0003.pdf`), reported that benchmarks were used,
and then stated plainly: *"It does not list the specific international institutions in
the provided pages."*

The retrieved chunks genuinely did not contain the institution list, so the judge
scored this `answered: false, grounded: false` — but the behavior is exactly right.
This is a **metric artifact, not a defect**: the groundedness rubric grades a declined
answer as ungrounded by construction. The true failure here is *retrieval* — the
institution names exist in the document (King Saud University, Carnegie Mellon, MIT and
others appear in the source) but did not surface in the top-5 chunks. Higher `top_k`
with reranking is the fix.

### 5. Abstention behavior is sound

All three out-of-scope questions were declined rather than answered from model
knowledge — including `oos-01`, where the corpus *mentions* GDPR but contains no fine
amounts. The system correctly distinguished "this topic appears" from "this question is
answerable," which is the harder discrimination.

## Threats to validity

**The groundedness figure is only meaningful after a harness fix.** An earlier run
scored it at 24.2%, contradicting a 97% retrieval hit rate and answers that were
verifiably correct against the source documents. The cause was in the harness, not the
system: the judge received each chunk truncated to 800 characters, so supporting text
was cut away before grading. A controlled comparison on `cls-01` confirmed it —
identical question and answer, judged `grounded: false` with truncated context and
`grounded: true` with full context. The harness now passes untruncated context, and
93.9% is the first valid measurement.

**Chunking cannot be credited for the improvement.** This run followed a re-ingest with
`SentenceSplitter(chunk_size=700, chunk_overlap=100)`, but the jump from 24.2% to 93.9%
is attributable to the judge fix, not to chunking. The chunking change was in fact
close to a no-op: `SimpleDirectoryReader` already emits one document per PDF page, and
those pages are mostly shorter than 700 tokens, so the splitter had little to split —
435 pages became 447 chunks (from 439 under default settings). Retrieval hit rate was
unchanged at 97.0% with the same single miss. Meaningfully changing chunk granularity
on this corpus requires either a much smaller `chunk_size` or merging pages before
splitting.

Further caveats:

- **Single run, no repetition.** Temperature is non-zero; per-question outcomes may vary
  between runs. No variance is reported.
- **Retrieval hit rate is a weak proxy for answer quality.** It confirms the right
  document was retrieved, not that the answer is correct. `ara-01` illustrates the gap
  in reverse: a wrong document produced a fluent answer.
- **`expected_sources` labels are author-assigned**, derived from document structure
  rather than annotated passages. A question may be legitimately answerable from a
  document not listed as expected.
- **Small out-of-scope sample.** 3 questions cannot characterize abstention robustly;
  100% here should be read as "no failures observed," not "will not fail."
- **The judge shares a model family with the system under test**, so correlated blind
  spots are possible.

## Reproducing

```bash
uv run python -m eval.run_eval             # full dataset + 5 agent runs
uv run python -m eval.run_eval --no-agent  # retrieval only, cheaper
uv run python -m eval.run_eval --limit 5   # smoke test
```

Results are written to `eval/results.json`, including every per-question answer,
retrieved source list, latency, and token count.
