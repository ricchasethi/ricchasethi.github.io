---
layout: post
title: "BioRAG: Part 4"
series: "BioRAG"
part: 4
date: 2026-05-27
---

# Building BioRAG: A Decision-Support RAG System for Biomedical Literature

**Part 4 of 5 — LLM Answer Synthesis, Prompt Design, and Serving the Engine**

---

## Where We Left Off

Post 3 ended with a fully assembled `DecisionOutput`.Confidence score derived from four explicit inputs, reasoning chain documenting every inference step, knowledge gaps surfaced as first-class fields. The rule-based `AnswerSynthesizer` concatenated excerpts, flagged contradictions with a warning marker, and found common terms across sources.

That is a valid baseline. It is auditable, dependency-free, and fast. But it does not write prose. For users who want natural language output, BioRAG provides `ClaudeSynthesizer` — a drop-in replacement that swaps only the prose generation step while leaving the rest of the pipeline identical.

This post covers three things: the design of `ClaudeSynthesizer` and the prompt choices that shaped it, the FastAPI server and the MCP server.

---

## The LLM Synthesizer

`ClaudeSynthesizer` is a **subclass** of `AnswerSynthesizer`, not a parallel implementation. It overrides exactly one method: `synthesize()`. The reasoning chain construction, confidence formula, gap detection, and follow-up generation are all inherited without modification.

```python
class ClaudeSynthesizer(AnswerSynthesizer):
    MODEL = "claude-sonnet-4-6"
    MAX_TOKENS = 1500
```

By the time a prompt reaches Claude, scoring, confidence computation, and gap detection have already run. Claude's only job is to write 2–4 paragraphs from the numbered excerpts it receives. This separation prevents the LLM from filling evidence gaps with training knowledge. This can be a problem that is much harder to enforce when the model owns the full pipeline.

The `anthropic` client is lazy-initialized so the import is never triggered for users running only the rule-based engine, preserving the core's zero-dependency guarantee.

---

## The System Prompt

The system prompt is a module-level constant in `llm_synthesizer.py` so it is easy to `grep`, version-control, and audit without reading the class.

```python
SYSTEM_PROMPT = """\
You are a biomedical decision-support assistant integrated into a Retrieval-\
Augmented Generation (RAG) system called BioRAG.

## Your task
Answer the researcher's question using ONLY the numbered evidence excerpts \
provided in the user message.

## Strict grounding rules
1. Every factual claim in your answer MUST be traceable to at least one of \
the supplied excerpts.  Cite sources inline as [N] where N is the excerpt number.
2. Do NOT use knowledge from your training data.  If the excerpts do not \
contain sufficient information, say:
   "The indexed corpus does not contain sufficient evidence to answer this \
question.  Consider ingesting additional papers on this topic."
3. If excerpts contradict each other, acknowledge the contradiction explicitly
and cite both sides.
4. Do not speculate, extrapolate, or infer beyond what the text states.

## Output format
Write 2–4 concise paragraphs of plain prose (no markdown headers, no bullet \
lists).  Begin with a direct answer to the question, then elaborate with \
supporting detail and citations.  End with a brief sentence noting any \
important limitations visible in the evidence.

## Evidence quality legend
- DIRECT — explicit statistical results or clinical findings relevant to the query.
- INDIRECT — topically related but does not directly confirm the query.
- CONTRADICTORY — contains language suggesting refutation or failure.
"""
```

### Why training knowledge is forbidden

An LLM trained on biomedical literature knows a great deal about Alzheimer's disease and pTau217. That knowledge is often correct. The problem is that it is not traceable. "pTau217 has an AUC of 0.94" means something different if it comes from a 2024 validation cohort of 800 patients than from a 2019 pilot of 40. The cohort characteristics and replication status are not knowable from a claim that originates in parametric memory.

### Why rule 4 exists

Rule 4 — "Do not speculate, extrapolate, or infer beyond what the text states" — was added after a failure case in development.

The query was *"What is the diagnostic performance of plasma pTau217 for Alzheimer's disease?"*. The first three paragraphs were correctly grounded: AUC values cited from [1] and [2], the contradictory replication finding from [3] acknowledged, community-based sample limitations noted. Then the model added:

> *"Given the trajectory of plasma biomarker development and declining cost of mass spectrometry platforms, pTau217-based testing is likely to become part of routine clinical practice within the next decade."*

No excerpt said that. The sentence was plausible and a knowledgeable reader might have agreed with it. Which is exactly what makes it dangerous. It was not marked as speculative. Detection was straightforward: `engine.synthesizer.last_prompt["user"]` showed the full excerpt set and none contained a forward-looking clinical practice claim.

The fix was rule 4 plus a phrasing change in the output format instruction. The original read "elaborate with supporting detail"; the revision reads "elaborate with supporting detail *and citations*", ensuring elaboration cannot license detail-addition without a citation anchor.

---

## The User Message

`_build_user_message()` produces the exact string that appears in the `messages` array:

```python
for i, e in enumerate(evidence, 1):
    support_label = e.support_type.upper()
    excerpt = TextProcessor.truncate_at_sentence(e.excerpt, 500)
    lines.append(
        f"\n[{i}] **{e.doc_title}** | Section: {e.section} | "
        f"Relevance: {e.relevance_score:.0%} | Type: {support_label}\n"
        f"Excerpt: {excerpt}"
    )
```

Three things in the excerpt header carry signal.

**Support type labels.** Each header includes `DIRECT`, `INDIRECT`, or `CONTRADICTORY`. The model already knows what these mean from the system prompt, so it knows before reading whether it is about to encounter a primary finding or a refutation — and can frame it accordingly in prose.

**Relevance percentages.** The reranker's score is passed through to the model. An excerpt at 64% should not be weighted the same as one at 100%. Providing the percentage means the model can structure its answer to lead with the strongest evidence.

**Gap list in the user message.** Gaps from `KnowledgeGapDetector` are listed in a "Known knowledge gaps" section. This prevents the model from treating absence of evidence as evidence of absence. If the gap list says "quantitative data not found", the model knows the corpus lacked the statistics, not that the statistics do not exist. It also makes the limitations sentence at the end of the answer specific rather than generic boilerplate.

The grounding instruction is repeated at the end of the user message ("Using ONLY the excerpts above...") as a recency anchor just before generation.

---

## Prompt Caching and Temperature

The system prompt is passed with `"cache_control": {"type": "ephemeral"}`. The cache TTL is 5 minutes. For a decision-support session where a clinician iterates through related queries, refining an initial question and following up on a gap — the system prompt tokens are processed once and cached for all subsequent queries in the session. At current pricing this reduces the effective per-query cost of the synthesis step by roughly 90% for those cached tokens.

Temperature is 0. Evidence synthesis in a clinical context has no use for sampling variation. The model should produce the same answer from the same evidence every time.

---

## Fallback on API Failure

If the API call fails, `ClaudeSynthesizer` falls back to the parent's rule-based `_build_answer()` and records it in the reasoning chain: `"LLM synthesis failed (...); used rule-based fallback."` The confidence score and gap list are unaffected. A decision-support system that becomes unusable when an external API is temporarily unavailable is not suitable for real use.

---

## The FastAPI Server

`server.py` exposes five endpoints:

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Corpus stats; confirms engine loaded |
| `POST` | `/query` | Full RAG pipeline; returns `QueryResponse` |
| `POST` | `/ingest` | Fetch PubMed/PMC papers and index them |
| `POST` | `/documents` | Add a single document by text |
| `GET` | `/corpus` | Document/chunk/term count statistics |

The query response includes `latency_ms` in the body so clients can detect when retrieval slows as the corpus grows. This a signal to tune `retrieval_top_k` or chunk size.

The `/ingest` endpoint is synchronous. It blocks until NCBI returns all results, which is fine for `max_results` values of 10–20 but will time out for large requests. The right fix is background ingestion with a polling endpoint which is exactly what the MCP server implements.

Every request and response type is a Pydantic model. `QueryResponse` mirrors `DecisionOutput` with the addition of `latency_ms`. This is a serialization contract: it prevents internal dataclass fields like `frozenset` from leaking into the HTTP response.

---

## The MCP Server

`mcp_server.py` exposes four tools callable directly inside a Claude Code session:

| Tool | What it does |
|------|--------------|
| `query(question)` | Full RAG pipeline — answer, confidence, evidence, gaps, follow-ups |
| `ingest(pubmed_query, max_results)` | Starts a background ingestion job; returns `job_id` immediately |
| `ingest_status(job_id)` | Polls the job; includes `corpus_stats` when done |
| `corpus_stats()` | Document/chunk/term counts |

### Background ingest

MCP tool calls run inside a conversational context. A tool that blocks for 30 seconds on NCBI responses would time out the transport. The solution is a job registry: `ingest()` starts a daemon thread and returns a `job_id` immediately. `ingest_status()` polls it.

```python
@mcp.tool()
def ingest(pubmed_query: str, max_results: int = 10) -> dict:
    job_id = uuid.uuid4().hex[:8]
    _jobs[job_id] = {"status": "running", ...}
    threading.Thread(target=_ingest_job, args=(job_id, pubmed_query, max_results),
                     daemon=True).start()
    return {"job_id": job_id, "status": "running", "message": "..."}
```

The polling pattern in a Claude Code session looks like:

```
Use biorag to ingest 20 papers on CRISPR cancer therapy
→ { "job_id": "a3f9c1d2", "status": "running" }

Use biorag ingest_status for a3f9c1d2
→ { "status": "done", "indexed": 18, "corpus_stats": { ... } }
```

`BioRAGEngine` carries an internal `threading.Lock` that serialises concurrent calls, so an ingest job running alongside a query is safe without additional locking at the server layer. A query during active ingestion sees the index in whatever state the lock was last released.

### Registration

```bash
pip install "mcp[cli]"
claude mcp add biorag python /absolute/path/to/mcp_server.py
claude mcp list   # → biorag: python ... ✓ Connected
```

The server starts as a child process over `stdio` transport. MCP tools are injected at session startup — restart Claude Code after first registration.

### Using the tools

Once registered, the tools are available in any Claude Code session by name. You address them in natural language and Claude Code calls them directly — no Python required.

**Query the indexed corpus:**

```
Use biorag to query: what plasma biomarkers predict Alzheimer's disease?
```

Claude Code calls `query()` and receives the full `DecisionOutput` — answer text, confidence label, evidence nodes with relevance scores, knowledge gaps, and follow-up questions. Each follow-up can be sent as the next query against the same live index.

**Ingest papers from PubMed:**

```
Use biorag to ingest 15 papers on plasma pTau217 Alzheimer's biomarkers
```

This returns a `job_id` immediately. Poll for completion:

```
Use biorag ingest_status to check job a3f9c1d2
```

When status is `"done"`, the response includes updated corpus stats — document count, chunk count, full-text vs abstract breakdown — confirming the papers landed before you query.

**Check what is indexed:**

```
Use biorag corpus_stats
```

Useful before a query session to confirm the corpus covers the topic. If `full_text_count` is low relative to `doc_count`, most documents are abstract-only — a signal that retrieval precision may be limited by shallow text.

**A typical session:**

```
1. corpus_stats          → 8 docs, 312 chunks, 4 full-text
2. ingest 20 papers on "plasma biomarkers Alzheimer diagnosis"
3. ingest_status <id>    → done, 18 indexed, 312 → 1047 chunks
4. query: what AUC values have been reported for pTau217?
5. query: are there contradictory findings on plasma NfL?
6. query: which biomarkers have been validated in community cohorts?
```

Papers ingested in step 2 are available to all subsequent queries without re-ingestion. Each query returns an independent `DecisionOutput` with its own confidence score and gap list.

---

## What the LLM Does and Does Not Own

A reasonable question: why not let the LLM run the whole pipeline? It knows biomedical text.

The structured pipeline does things the LLM cannot do reliably. BM25 scoring is exact arithmetic over a data structure: a token either matches or it does not. The confidence formula produces a checkable number: `0.3 + 0.4 + 0.14 − 0.05 = 0.81` is verifiable by a human reviewer; "High confidence" produced by an LLM is not. The gap detector fires on structural properties of the evidence set that do not require reading comprehension, and its false-positive rate is knowable from the pattern definitions.

What the LLM does well — and what the structured components do poorly — is *writing*. `_build_answer()` concatenates excerpts. `ClaudeSynthesizer` compares them, qualifies them, and acknowledges contradictions in context. That is the right division of labour: deterministic structure where auditability matters, natural language where readability matters.

---

## Coming Up in Post 5

Post 5 covers the retrieval evaluation harness — the `evals/` directory that measures BM25 and reranker quality independently using MRR and NDCG@K. We will look at the 16-query ground truth set, how max-pooling aggregates chunk-level scores to document-level, and what the Δ column in the eval output actually tells you — when a negative NDCG delta is a reranker bug versus an expected trade-off from tightening `rerank_top_k`. We will also look at what retrieval evals cannot measure and what a full answer-quality evaluation would require.

---

*BioRAG is built in pure Python stdlib. The core engine — BM25 index, reranker, evidence classifier, synthesizer — has zero external dependencies. PubMed ingestion requires only `requests`. LLM answer synthesis requires only `anthropic`.*
