---
layout: post
title: "BioRAG: Part 8"
series: "BioRAG"
part: 8
date: 2026-06-24
---

# Building BioRAG: A Decision-Support RAG System for Biomedical Literature

**Part 8: Does Hybrid Retrieval Actually Help? Measuring It on Alzheimer's Queries**

---

In Part 7, I added a second way to search BioRAG's corpus. Alongside the original keyword search (BM25), the engine can now search by *meaning* using embeddings stored in a vector database, and it merges the two result lists into one.

That post made the case for *why* this should help. This post asks the only question that matters: **does it?** 

The short answer: **for Alzheimer's queries, the full hybrid pipeline lifts the headline metric by about 15%.** The long answer is more interesting than that number because the gain does not come from where you would expect.

---

## A One-Minute Refresher on the Metrics

Part 5 covered these in depth: here is just enough to read the tables.

- **MRR@5** (Mean Reciprocal Rank) — *how high does the first correct document land?* If the right document is at rank 1, that query scores 1.0; at rank 2, it scores 0.5; at rank 3, 0.33; if it is nowhere in the top 5, it scores 0. Average across all queries. **Higher is better.** This is the headline number for decision-support, because a clinician reads from the top and stops at the first useful answer.
- **NDCG@K** (Normalized Discounted Cumulative Gain) — *is the whole ranked list well-ordered?* It rewards putting highly relevant documents above partially relevant ones, and penalizes burying a good document deep in the list. Also 0 to 1, higher is better.

Both measure **retrieval**. Whether the right document was *found and ranked well*. They say nothing about whether the final answer was any good. That was Part 6's job.

---

## Four Searches, Not Two

It is not enough to compare "old" against "new" to find whether hybrid retrieval helps or not. That would tell me the combined system changed but not *which part* caused the change. So the eval harness (`evals/retrieval_eval.py --hybrid`) measures **four** retrieval modes independently on the same queries:

| Mode | What it does |
|---|---|
| **BM25** | Keyword search only — the original system from Parts 1–6 |
| **Dense** | Embedding search only — the new semantic retriever, with BM25 turned off |
| **Hybrid** | BM25 + Dense, merged — but *before* the reranker runs |
| **Hybrid+Rerank** | The full new pipeline — merged candidates, then reranked |

This is the same philosophy as the BM25-vs-reranker comparison in Part 5: isolate each stage so any change is *attributable*. If the full pipeline improves, these four columns tell me whether the credit belongs to embeddings, to the merge step, or to the reranker doing a better job because it had more to work with.

### How the merge works: Reciprocal Rank Fusion

When BM25 and the embedding search each return a ranked list, I need to combine them into one. The two scores are not comparable as BM25 produces unbounded magnitudes and cosine similarity lives between -1 and 1. So, I cannot just add them.

The trick is to **ignore the scores and use only the rank positions**. This is **Reciprocal Rank Fusion (RRF)**. Each document gets a fused score:

```
fused_score(doc) = Σ   1 / (k + rank_of_doc_in_that_list)
                 lists
```

with `k = 60` (a standard default). A document ranked 1st contributes `1/61`; ranked 2nd, `1/62`; and so on. A document that appears near the top of **both** lists accumulates two contributions and rises; a document near the top of only one list still gets a meaningful boost. Because only rank position matters, RRF sidesteps the incompatible-scales problem entirely. Documents missing from a list simply contribute nothing from it.

---

## The Headline Result for Alzheimer's

Here is the eval run on the seven Alzheimer's disease queries (`--hybrid --alzheimer-only`):

```
  Metric                BM25           Dense          Hybrid   Hybrid+Rerank
  ──────────────────────────────────────────────────────────────────────────
  MRR@5                0.714           0.719           0.695           0.821
  NDCG@1               0.571           0.571           0.571           0.714
  NDCG@3               0.752           0.733           0.714           0.804
  NDCG@5               0.752           0.788           0.770           0.866
  Δ vs BM25 (MRR@5):  Dense +0.005  Hybrid -0.019  Hybrid+Rerank +0.107
```

Read the last row first. Against the BM25 baseline:

- **Dense alone: +0.005.** Essentially a tie. Pure semantic search, on average, is no better than keyword search here.
- **Hybrid (merge, no rerank): −0.019.** **Worse.** Merging the two lists actually *hurt* the headline metric.
- **Hybrid+Rerank: +0.107.** The full pipeline lifts MRR@5 from 0.714 to 0.821 — a **15% relative improvement** — and lifts NDCG@5 from 0.752 to 0.866.

So the answer is *yes, it helps*. But the story is not "embeddings are better than keywords". Neither dense alone nor the merge alone improved anything. **The entire gain materializes only when the reranker runs on top of the fused candidate set.** That deserves an explanation.

---

## Why the Gain Hides Until the Reranker Runs

This is the most important finding in the post, so it is worth slowing down.

Think of retrieval as two jobs: **recall** (getting the right documents *somewhere* in the candidate pile) and **precision** (ordering that pile so the best one is on top).

- **Embeddings improve recall.** They drag in relevant documents that BM25 missed because of paraphrasing. But the raw fused ordering is mediocre because RRF orders by a crude rank-position formula that knows nothing about biomedical sections, entities, or which chunk actually contains the result.
- **The reranker improves precision.** BioRAG's reranker (Part 2) understands section structure and applies a discriminative-token penalty to suppress off-topic matches. On its own it cannot retrieve a document that was never fetched. But give it a *richer, more complete* candidate pool, and it has better raw material to promote the truly relevant document to rank 1.

That is exactly the pattern in the numbers. The merge step (Hybrid) widens the candidate pool that improves recall but slightly scrambles the order. So, MRR dips to 0.695. Then the reranker, now choosing from that wider pool, sorts it properly and pushes MRR up to 0.821. **Fusion supplies the candidates; the reranker converts them into rank-1 hits.** Neither half does the job alone.

Notice NDCG@1 too: it jumps from 0.571 to 0.714 only in the Hybrid+Rerank column. NDCG@1 measures "is the single top result the best possible document?" and that is precisely the precision job the reranker performs on the enriched pool.

---

## Per-Query: Where Embeddings Win, and Where They Hurt

Averages hide the mechanism. The per-query MRR@5 breakdown is where you actually see embeddings earning and losing:

```
  Query                  BM25     Dense   Hybrid  Hybrid+Rerank
  ──────────────────────────────────────────────────────────────
  Q01 (diagnosis)        1.000     0.333    0.333      1.000
  Q02 (diagnosis)        1.000     1.000    1.000      1.000
  Q03 (diagnosis)        0.500     0.200    0.333      0.250
  Q04 (comparison)       0.000     0.500    0.200      0.500
  Q05 (prognosis)        1.000     1.000    1.000      1.000
  Q06 (mechanism)        0.500     1.000    1.000      1.000
  Q07 (treatment)        1.000     1.000    1.000      1.000
```

Three of these queries (Q02, Q05, Q07) are already solved by every method (all four columns score 1.000), so there is no room to improve them. The action is in the four queries where the methods disagree: Q01, Q03, Q04, and Q06.

**Q06 (mechanism): the textbook win for embeddings.**
BM25 scores 0.500 (right document at rank 2); Dense, Hybrid, and Hybrid+Rerank all score 1.000 (rank 1). Mechanism questions ("how does X cause Y?") are phrased abstractly and rarely repeat the document's exact vocabulary. Semantic search closes it cleanly. **This single query is the clearest evidence that embeddings retrieve something keywords cannot.**

**Q04 (comparison): embeddings rescue a total miss.**
BM25 scores **0.000**. It failed to surface the relevant document anywhere in the top 5. Dense pulls it to 0.500 and Hybrid+Rerank holds it at 0.500. A complete keyword failure turned into a rank-2 hit purely because semantic search understood what the comparison was *about*. When BM25 returns nothing useful, embeddings are the safety net.

**Q01 (diagnosis): the cautionary tale.**
BM25 nails it at 1.000. Dense **collapses to 0.333** and so does the raw Hybrid merge. For a precise diagnostic query full of exact biomarker names, the fuzzy semantic search actively *hurt* — it dragged in topically-similar-but-wrong documents and demoted the exact match. **But Hybrid+Rerank recovers it back to 1.000**, because the reranker's discriminative-token penalty recognizes the exact-entity match and restores it to the top. This is the single best argument for *hybrid* over *dense-only*: the reranker repairs the damage that pure semantic search does to precise queries.

**Q03 (diagnosis): an honest regression.**
Here there is no happy ending. BM25 scores 0.500; Hybrid+Rerank scores **0.250** — worse. The relevant document slips from rank 2 to rank 4. Hybrid retrieval is not a free lunch: on some queries the extra candidates crowd out a document BM25 had ranked reasonably. One regression out of seven is a real cost and it is the kind of thing that only shows up if you measure per-query instead of trusting the average.

---

## The Full 16-Query Picture

Zooming out to all 16 queries across every disease area (`--hybrid`, no subset filter):

```
  Metric                BM25           Dense          Hybrid   Hybrid+Rerank
  ──────────────────────────────────────────────────────────────────────────
  MRR@5                0.875           0.877           0.867           0.922
  NDCG@1               0.771           0.771           0.812           0.875
  NDCG@3               0.869           0.854           0.847           0.886
  NDCG@5               0.869           0.878           0.878           0.913
  Δ vs BM25 (MRR@5):  Dense +0.002  Hybrid -0.008  Hybrid+Rerank +0.047
```

The same shape, muted. Hybrid+Rerank still wins (MRR@5 0.875 → 0.922; NDCG@1 0.771 → 0.875), and the merge-without-rerank step still dips slightly negative (−0.008). The improvement is smaller than on the Alzheimer's subset (+0.047 vs +0.107) for a simple reason: **most of the 16 queries already score a perfect 1.000 under plain BM25** (thirteen of sixteen). There is no headroom to improve a query that is already perfect, so the average gain is diluted by easy wins.

This is actually the central lesson about *where* hybrid retrieval pays off: **the gains concentrate entirely on the hard queries.** The Alzheimer's subset contains a higher proportion of genuinely difficult retrievals like the paraphrased mechanism query (Q06), the keyword-failure comparison (Q04). This is why the improvement there is more than double the full-set average. Hybrid retrieval does not make easy queries easier. It rescues the hard ones.

---

## So — Does It Improve Alzheimer's Retrieval?

**Yes, clearly — but with three honest qualifications:**

1. **The full pipeline wins.** Hybrid+Rerank improves MRR@5 by 15% (0.714 → 0.821) and NDCG@5 by 15% (0.752 → 0.866) on Alzheimer's queries. The biggest single jump is NDCG@1 (0.571 → 0.714).
2. **The credit goes to recall + reranking, not to embeddings alone.** Dense-only and merge-only each improved nothing on average. Embeddings widen the candidate pool (rescuing Q04 and Q06); the reranker turns that wider pool into rank-1 hits and repairs the damage dense search does to precise queries (Q01). Remove either half and the gain disappears.
3. **It is not uniformly positive.** Pure semantic search can badly hurt precise diagnostic queries (Q01: 1.000 → 0.333), and even the full pipeline regressed one query (Q03: 0.500 → 0.250). The net effect is a clear win, but it is a *distribution* of wins and losses, not a guarantee on every query.

For a decision-support system, qualification 2 is the design takeaway. Adding embeddings is not a drop-in upgrade that "makes search smarter." It is half of a system whose other half is what actually converts richer candidates into better answers. Ship the embeddings without the reranker and, on this corpus, you would have *degraded* the headline metric.

---

## Reproducing This

Every number above comes from one command:

```bash
# Alzheimer's subset (7 queries), per-query detail
python evals/retrieval_eval.py --hybrid --alzheimer-only --verbose

# Full 16-query set
python evals/retrieval_eval.py --hybrid --verbose
```

The harness builds an in-memory vector index so the eval always reflects the current corpus, embeds every chunk with the PubMed-tuned model, and runs all four modes through the same metric functions used in Part 5. The four-mode comparison is the retrieval analogue of Part 6's rule-based-vs-LLM answer-quality table: it makes the contribution of each stage visible and attributable, so you are never improving a black box.

---

## Takeaways

- **Hybrid retrieval improves Alzheimer's queries by ~15% on the headline metric** — a real, measured gain, not a theoretical one.
- **The gain is a team effort.** Embeddings improve *recall* (finding paraphrased and keyword-missed documents); the reranker improves *precision* (ordering the enriched pool). Each alone improved essentially nothing on average.
- **Merging two retrievers can slightly hurt ranking before reranking** — the candidate pool gets richer but messier. Always measure the stage *after* the one you changed.
- **Gains concentrate on hard queries.** On the full set, where most queries already score 1.0, the improvement is smaller. This is because hybrid retrieval rescues the difficult retrievals rather than improving the easy ones.
- **Measure per-query, not just the average.** The average said "+0.107." The per-query view revealed an embeddings triumph (Q06), a keyword rescue (Q04), a precision near-disaster repaired by reranking (Q01), and one genuine regression (Q03). Only one of those four stories survives in the mean.

The honest verdict: hybrid retrieval is worth shipping for biomedical decision-support.

---

*Code: [github.com/ricchasethi/rag-biomedical-decision](https://github.com/ricchasethi/rag-biomedical-decision)*
