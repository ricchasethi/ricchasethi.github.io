---
layout: post
title: "BioRAG: Part 5"
series: "BioRAG"
part: 5
date: 2026-06-01
---

# Building BioRAG: A Decision-Support RAG System for Biomedical Literature

**Part 5 of 5 — Evaluation, MRR, NDCG, and Lessons from Building Retrieval Evals**

---

## Where We Left Off

The previous four posts built the pipeline. Post 1 covered query analysis and BM25 retrieval. Post 2 covered reranking and evidence classification. Post 3 covered confidence scoring and gap detection. Post 4 covered LLM synthesis and serving.

None of those posts asked the hardest question: does any of it actually work?

"Works" in retrieval has a precise meaning. The issue is system always returns results even from an irrelevant corpus. It means the results that matter are ranked near the top, and the results that do not matter are ranked below them. Measuring that precisely is what the `evals/` harness is for.

---

## What the Harness Measures

The harness evaluates two pipeline stages independently:

- **Stage A — BM25 alone.** The `InvertedIndex` retrieves the top-60 chunks by raw BM25 score, aggregated to document level.
- **Stage B — BM25 + Reranker.** The same BM25 output is passed through `Reranker.rerank()`, producing a top-5 document list.

Comparing them isolates the reranker's contribution. If Stage B is better, the reranker is earning its place. If Stage B is worse, the reranker is introducing a regression. With a single aggregate metric across the whole pipeline, you cannot tell which stage caused a change. With two independent measurements, you can.

The evaluator bypasses the full `engine.query()` path and calls components directly:

```python
q_analysis = self.engine.query_analyzer.analyze(rq.query)

# Stage A
bm25_chunks = self.engine.index.search(q_analysis["expanded_tokens"], top_k=60)
bm25_doc_ranking = self._chunks_to_doc_ranking(bm25_chunks)

# Stage B
reranked_chunks = self.engine.reranker.rerank(bm25_chunks, q_analysis, top_k=5)
reranked_doc_ranking = self._chunks_to_doc_ranking(reranked_chunks)
```

The evidence classifier and synthesizer are not involved. They operate on the output of retrieval; they do not change retrieval quality and should not contaminate the measurement.

---

## Chunk-to-Document Aggregation

BM25 retrieves chunks. The ground truth labels documents. Bridging that gap requires a choice.

The harness uses **max-pooling**: group all chunks by `doc_id`, take the highest-scoring chunk per document, and sort documents by that maximum.

```python
def _chunks_to_doc_ranking(self, chunks: list[RetrievedChunk]) -> list[tuple[str, float]]:
    doc_scores: dict[str, float] = {}
    for rc in chunks:
        doc_id = rc.chunk.doc_id
        doc_scores[doc_id] = max(doc_scores.get(doc_id, 0.0), rc.score)
    return sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
```

The alternative is mean-pooling: average all chunk scores for a document. Mean-pooling is appropriate when relevance is distributed evenly across a document's content. For biomedical papers, it is not. A paper's key finding might appear in one Results-section chunk with a high BM25 score. And every other chunk from the same paper is methods boilerplate or background text with low scores. Mean-pooling dilutes the strong signal with noise. Max-pooling treats one excellent chunk as sufficient evidence that the document is relevant.

**Why `_bm25_top_k = 60`:** The evaluator retrieves 60 chunks at the BM25 stage, far more than the engine's default `retrieval_top_k=15`. With a 4-document corpus and roughly 10 chunks per document, capping at 15 would leave documents unrepresented in the BM25 ranking. The eval harness needs every document to have a score, so the ceiling is set generously. Production retrieval does not need this — it is eval-specific.

---

## The Ground Truth

`evals/ground_truth.py` defines 16 `RetrievalQuery` objects with hand-labelled `relevant_docs` dicts. Each entry maps a `doc_id` to a relevance grade:

| Grade | Meaning |
|-------|---------|
| `2` | Highly relevant — the document directly answers the question |
| `1` | Partially relevant — related but tangential |
| absent | Irrelevant (grade 0) |

The corpus has four documents: a cardiovascular/diabetes paper, an oncology paper, an infectious disease paper, and an Alzheimer's biomarker review. The 16 queries are split across five sub-lists:

```python
EVAL_QUERIES = (
    ALZHEIMER_QUERIES   # Q01–Q07 — seven queries, one relevant doc each
    + CARDIO_QUERIES    # Q08–Q10 — three queries, one or two relevant docs
    + ONCO_QUERIES      # Q11–Q12 — two comparison/treatment queries
    + INFECT_QUERIES    # Q13–Q14 — two treatment queries
    + CROSS_QUERIES     # Q15–Q16 — cross-topic discrimination queries
)
```

Q15 is the most demanding query in the set:

```python
RetrievalQuery(
    query_id="Q15",
    query="What systematic review evidence exists for biomarker-guided therapy in complex diseases?",
    intent="general",
    relevant_docs={
        "neuro_2026_001": 2,   # umbrella review of biomarker systematic reviews
        "cardio_2026_001": 1,  # RCT meta-analysis in diabetes
        "onco_2026_001": 1,    # biomarker-guided therapy in NSCLC
    },
)
```

Three documents are relevant at different grades. A system that retrieves all three but ranks the oncology paper first (because "biomarker-guided" appears prominently in its abstract) scores lower than one that ranks the Alzheimer's review first. Q15 tests whether the system can discriminate primary relevance from partial relevance when multiple documents match the query's generic vocabulary.

Q16 is the discrimination test: a query about drug-resistant bacterial infections should surface `infect_2026_001` at rank 1 and push the three unrelated papers below it, even though all four papers contain the word "treatment" and generic disease vocabulary.

---

## The Metrics

### MRR (Mean Reciprocal Rank)

MRR measures how quickly the system surfaces the **first** relevant result. For each query, find the rank of the first relevant document in the top-K list. Its reciprocal rank is `1/rank`. MRR is the mean over all queries.

```python
def reciprocal_rank(doc_ranking, relevant_docs, k) -> float:
    for rank, (doc_id, _score) in enumerate(doc_ranking[:k], start=1):
        if relevant_docs.get(doc_id, 0) > 0:
            return 1.0 / rank
    return 0.0
```

A relevant document at rank 1 contributes 1.0. At rank 2, it contributes 0.5. At rank 3, 0.33. MRR falls to 0 if no relevant document appears in the top-K.

**Why MRR for decision support:** A clinician using BioRAG reads results until they find a useful answer. They stop at the first one that answers their question. MRR penalises a system that buries the answer at rank 3 twice as harshly as one that surfaces it at rank 2. That penalty structure matches the actual cost of a bad ranking in a clinical use case.

### NDCG (Normalised Discounted Cumulative Gain)

NDCG accounts for the full ranked list and the graded relevance distinctions. Its formula:

```
DCG@K  = Σ (2^rel_i − 1) / log2(i + 1)   for i = 1 … K
IDCG@K = DCG of the ideal (perfect) ordering
NDCG@K = DCG@K / IDCG@K
```

The exponential gain formula `2^rel − 1` matters:
- Grade 2 → gain of 3
- Grade 1 → gain of 1
- Grade 0 → gain of 0

A grade-2 document contributes three times as much as a grade-1 document at the same rank. For Q15, correctly placing the Alzheimer's review at rank 1 matters far more than whether the oncology paper appears at rank 2 or rank 3.

The log2 discount means rank 1 is worth roughly 3× rank 3. A system that places the grade-2 document at rank 3 instead of rank 1 scores substantially lower, even though it technically "found" the document.

**K values evaluated: 1, 3, and 5.**

- K=1 — did the top result hit?
- K=3 — quality of the first screen of results, which is what a user typically reviews
- K=5 — covers all four corpus documents, so it acts as a global ranking quality ceiling

For a 4-document corpus, NDCG@5 measures how well the system ordered all documents relative to the ideal ordering. For production use where the corpus is thousands of documents, K=5 is far from a ceiling. But the K=3 signal remains the most actionable: if your reranker hurts NDCG@3, users are getting worse results on their first screen.

---

## Reading the Output

The report is structured as a summary table, a by-intent breakdown, and optional per-query detail:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  BioRAG Retrieval Eval  (16 queries)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Metric              BM25    Reranked         Δ
  ────────────────────────────────────────────────────────────
  MRR@5              0.863       0.906      +0.043
  NDCG@1             0.813       0.875      +0.063
  NDCG@3             0.844       0.891      +0.047
  NDCG@5             0.876       0.882      +0.006

  Note: Δ = Reranked − BM25  (positive = reranker improves ranking)

  By Intent           N    BM25 MRR   Reranked       Δ
  ────────────────────────────────────────────────────────────
  comparison          3       0.833      0.833      +0.000
  diagnosis           4       0.875      0.938      +0.063
  epidemiology        1       1.000      1.000      +0.000
  general             1       0.750      0.750      +0.000
  mechanism           1       1.000      1.000      +0.000
  prognosis           1       1.000      1.000      +0.000
  treatment           5       0.800      0.860      +0.060
```

### What the Δ column reveals

A positive Δ means the reranker improved ranking over raw BM25. A negative Δ means it introduced a regression. The Δ column is the primary diagnostic signal.

**When a negative NDCG@3 Δ is a reranker bug:** If a query with a single grade-2 relevant document shows BM25 placing it at rank 2 and the reranker demoting it to rank 4, that is a section weighting or discriminative-token penalty misfiring. Diagnosis: run `--verbose` to see the per-query top-3 for both stages, identify which document moved and why. Check whether its section header was detected correctly (a `Body`-classified Results section receives no section boost). Check whether any of its key terms accidentally landed in `_GENERIC_BIO_TERMS` and were excluded from the discriminative set.

**When a negative NDCG@5 Δ is expected:** With `rerank_top_k=5` and a corpus of 4 documents, the reranker is not cutting any documents — they all fit within the budget. For a larger corpus where `rerank_top_k` is genuinely constraining, a negative NDCG@5 Δ can appear even when NDCG@3 is positive. This happens when BM25 had a grade-1 document at rank 6 that the reranker deprioritised below rank 5, improving the top-3 precision but losing the grade-1 document from the scored window. This is not a bug — it is the reranker doing its job. The question is whether the trade-off is worth it for your use case. If users only review the first three results, a positive NDCG@3 and negative NDCG@5 is a net improvement.

The by-intent breakdown is where the most actionable signal lives. The `comparison` intent shows Δ = 0.000 — the reranker is neither helping nor hurting. That is a candidate for tuning: `SECTION_WEIGHTS["comparison"]` currently weights Results at 1.4× and Discussion at 1.2×. If `--verbose` shows comparison queries retrieving Discussion chunks first, the Results weight may need raising or section detection for that document type may need fixing.

---

## Running the Harness

```bash
# All 16 queries
python evals/retrieval_eval.py

# Alzheimer's subset only (Q01–Q07)
python evals/retrieval_eval.py --alzheimer-only

# Per-query top-3 detail vs ground truth
python evals/retrieval_eval.py --verbose

# Custom K values
python evals/retrieval_eval.py --ks 1 5 10
```

The `--verbose` flag is the most useful for debugging. Each row shows the top-3 retrieved documents for both BM25 and Reranked stages alongside the ground truth. When a document that should be at rank 1 appears at rank 3, the verbose output makes the failure obvious at a glance.

**Adding new queries to the ground truth** is the main maintenance task after corpus changes. The pattern is:

```python
RetrievalQuery(
    query_id="Q17",
    query="What is the role of APOE4 in Alzheimer's disease risk?",
    intent="mechanism",
    relevant_docs={
        "neuro_2026_001": 2,
        "new_paper_id": 1,
    },
),
```

Add it to the appropriate sub-list and it is automatically included in `EVAL_QUERIES`. The `query_id` is a stable identifier so results are traceable across runs. If NDCG@3 drops after a corpus change, you can run `--verbose` and filter on the affected `query_id` to see what changed.

---

## What Retrieval Evals Cannot Measure

The harness measures one thing: does the right document appear near the top of the ranked list? That is necessary but not sufficient for a functioning decision-support system.

**Retrieval quality ≠ answer quality.** A system can rank the correct document at rank 1 and still produce a misleading answer if the synthesizer misrepresents the evidence, the evidence classifier mislabels a contradictory finding as direct, or the knowledge gap detector fails to fire when it should. These failures are invisible to MRR and NDCG because they happen downstream of retrieval.

**Ground truth is expensive.** The 16 queries in this harness took time to label. Each `relevant_docs` dict is a human judgement call: is this document grade 2 or grade 1 for this specific query? For a production corpus of hundreds of documents and a realistic query set of 50+ queries, building a comprehensive ground truth is a significant investment that is often skipped. This is why retrieval evals are so frequently absent from RAG systems that could benefit from them.

**The eval corpus is small.** Four documents means the harness is sensitive to individual document idiosyncrasies. If `neuro_2026_001` happens to contain an unusual amount of repetition of a query term, a BM25 tuning change that helps all other queries may hurt Alzheimer's queries simply because that document's scoring is outlier-shaped. With four documents, you cannot tell the difference between a systematic improvement and an improvement that happened to benefit the only relevant document for 7 of 16 queries.

---

## What a Full Answer-Quality Evaluation Would Require

Retrieval evals measure where documents land in a ranked list. Answer-quality evals measure whether the final answer is correct and whether the confidence score is calibrated.

A calibrated confidence score has a specific meaning: if the system assigns confidence 0.81 to a set of answers, approximately 81% of those answers should be correct. Without a labelled answer-quality dataset and a correctness measure, you cannot verify this.

Building that evaluation requires three things retrieval eval does not:

1. **A set of queries with known correct answers.** For biomedical questions this is hard. The "correct" answer is often a range, depends on population, or is itself contested in the literature. Factoid questions (survival rates, AUC values for specific biomarkers) are easier to evaluate than mechanism questions.

2. **A correctness judge.** Human annotation is gold standard but slow. A model-based judge (another LLM assessing whether the answer matches the known correct answer) is faster but introduces its own errors. Whatever method you choose, it needs to be specified clearly enough to be reproducible.

3. **Calibration curve measurement.** Run the full pipeline on all eval queries. Bin the results by confidence score. Measure the fraction of correct answers in each bin. Plot confidence vs correctness rate. If the curve is monotonically increasing and close to the diagonal, confidence is calibrated. If high-confidence answers are correct only 60% of the time, the formula coefficients need adjustment.

BioRAG's confidence formula is conservatively designed: the base of 0.3 and the cap at 0.95 mean it defaults to skepticism. Calibration would require measurement against labelled outcomes, which requires an answer-quality harness that this codebase does not currently have. Building even a small one with 20–30 labelled queries would be the highest-value next step for making the confidence scores clinically trustworthy.

---

## Lessons from Building Retrieval Evals

A few things that are not obvious until you build one.

**Evaluate stages not the pipeline.** A single metric over the whole pipeline hides which stage is responsible for a change. Two independent measurements — BM25 alone, then BM25 + reranker — make regressions attributable and improvements verifiable. The cost is two evaluation passes per query; the benefit is that every tuning change to the reranker has an immediate isolated signal.

**Max-pooling is almost always right for chunk-level retrieval.** Mean-pooling sounds more principled but penalises documents that have one excellent chunk and many mediocre ones, which describes most biomedical papers with focused Results sections. If you are unsure, instrument both and compare against your ground truth. Max-pooling wins in every corpus we have tested with this pattern.

**NDCG@3 is the number to optimise.** K=1 is too sensitive to a single missed retrieval. K=5 (or higher) includes results the user will never read. K=3 corresponds to the first screen of results and is where the retrieval-quality signal is most actionable.

**The by-intent breakdown is where tuning lives.** An aggregate MRR of 0.906 looks good. An intent breakdown that shows comparison queries at Δ = 0.000 and treatment queries at Δ = +0.060 tells you exactly where to tune next. Aggregate numbers hide which query types the system handles well and which it does not.

**Ground truth rots.** After ingesting new papers, the relevant document set for existing queries changes. A query about Alzheimer's biomarkers that had one grade-2 document now has two, but only if you update the ground truth. Stale ground truth produces misleading improvements: a new paper that should be grade-2 for seven queries is invisible to the eval until you add it. Treat ground truth maintenance as a first-class task, not an afterthought.

---

## The Series So Far

Five posts, one pipeline. Here is the full map:

| Post | Stage | Key idea |
|------|-------|----------|
| 1 | QueryAnalyzer + BM25 | Exact token matching; discriminative signal in biomedical text lives in rare terms |
| 2 | Reranker + EvidenceClassifier | Section weighting + discriminative-token recall penalty suppress multi-disease false positives |
| 3 | KnowledgeGapDetector + AnswerSynthesizer | Confidence is a formula with explicit inputs; calibrated uncertainty is a feature |
| 4 | ClaudeSynthesizer + FastAPI + MCP | LLM owns only prose generation; structured pipeline owns everything auditable |
| 5 | Retrieval eval harness | Measure stages independently; retrieval quality and answer quality are different problems |

BioRAG is not a finished product. The discriminative-token set is hand-curated and needs to become corpus-derived. Section detection misses non-standard headers. The confidence formula is conservative but uncalibrated. The answer-quality harness does not exist yet.

But each of those limitations is documented, localised to a specific component, and addressable without touching the rest of the pipeline. That is what the single-responsibility architecture was designed to make possible.

---

*BioRAG is built in pure Python stdlib. The core engine — BM25 index, reranker, evidence classifier, synthesizer — has zero external dependencies. PubMed ingestion requires only `requests`. LLM answer synthesis requires only `anthropic`.*
