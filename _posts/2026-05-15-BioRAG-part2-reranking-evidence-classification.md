---
layout: post
title: "BioRAG"
series: "BioRAG"
part: 2
date: 2026-05-15
---

# BioRAG

**Part 2 of 5 — Reranking, Evidence Classification, and False Positive Suppression**

---

## Where We Left Off

Post 1 ended with 15 BM25 candidates sitting in a list. Most were good. One was a lung cancer paper that had muscled its way into rank 9 of an Alzheimer's disease query. The culprit was legitimate BM25 scoring. It used "biomarker" and "disease" heavily. It never mentioned Alzheimer's once.

That paper is the problem this post solves.

By the end of Post 1, the pipeline had produced this:

```python
retrieved_chunks = [
    RetrievedChunk(doc_title="Plasma Aβ42/p-Tau217 ratio...",       score=9.72, rank=1),
    RetrievedChunk(doc_title="Diagnostic Blood-Based Biomarkers...", score=8.10, rank=2),
    RetrievedChunk(doc_title="EGFR-Mutant Non-Small Cell Lung...",   score=7.30, rank=9),
    # ...
]
```

This post covers the two stages that turn those 15 raw candidates into 5 high-quality evidence nodes: the **Reranker** and the **EvidenceClassifier**.

---

## The Reranker: Three Signals on Top of BM25

The `Reranker` does not replace BM25. It applies three additional signals to adjust the scores BM25 already produced and keeps only the top-K results. The three signals are section weighting, term density, and discriminative-token recall.

### Signal 1: Section Weighting

Not all sections of a biomedical paper are equally valuable for a given query intent. A mechanism query wants the Methods and Results sections — those sections contain the pathway experiments and quantitative findings. A definition query wants the Introduction or Abstract. A treatment query wants Results and Discussion.

The `Reranker` encodes this as a lookup table:

```python
SECTION_WEIGHTS = {
    "mechanism": {"Results": 1.3, "Methods": 1.1, "Discussion": 1.0, "Abstract": 0.9},
    "comparison": {"Results": 1.4, "Discussion": 1.2, "Abstract": 1.0},
    "treatment":  {"Results": 1.4, "Methods": 1.2, "Discussion": 1.0},
    "diagnosis":  {"Results": 1.3, "Methods": 1.2, "Abstract": 1.0},
    "definition": {"Introduction": 1.3, "Abstract": 1.2, "Body": 1.0},
    "general":    {"Abstract": 1.1, "Results": 1.1, "Discussion": 1.0},
}
```

Each chunk already carries a `section` field that is set at chunking time by `DocumentChunker._detect_section()`. This scans the first line of each chunk for header patterns. A chunk from a Results section with a diagnosis query gets a 1.3× multiplier. A chunk from an undetected section falls back to 1.0.

**What this buys you:** For a query like "What plasma biomarkers detect Alzheimer's disease?", a Results chunk reporting AUC values for pTau217 starts with a 1.3× advantage over an Introduction chunk summarising background literature. The Introduction chunk may contain every query term; the Results chunk contains the answer.

**What this does not solve:** Section detection is heuristic. A paper whose Results section is titled "Findings" or "Outcomes" will not be detected — those chunks land in `Body` at 1.0. For multi-disease corpora sourced from diverse journals, section header conventions vary. If section weights are not engaging, check chunk-level `section` distributions first.

### Signal 2: Term Density

Term density is the fraction of a chunk's tokens that are query tokens:

```python
chunk_token_set = set(r.chunk.tokens)
if chunk_token_set:
    density = len(query_tokens & chunk_token_set) / len(chunk_token_set)
```

A chunk where 15% of all tokens are query terms is denser and likely more focused than one where the same five query terms appear across 500 tokens of surrounding context. The density contributes a `(1 + density × 0.4)` multiplier, capping at a 40% bonus for a perfectly dense chunk.

This is a mild signal. Its main job is to break ties between two chunks from the same section of the same paper. A shorter, more focused chunk should win over a longer one that matches the same terms but buries them in noise.

### Signal 3: Discriminative-Token Recall

This is the signal that matters most and the one that requires the most explanation.

#### The Root Problem

BM25 scores a chunk based on the terms it *contains*. It has no concept of terms a chunk is *missing*. This is fine for general-domain retrieval. In a multi-disease biomedical corpus, it is a structural problem.

Consider a query: *"What plasma biomarkers predict Alzheimer's disease?"*

After tokenisation and stopword removal, the query produces tokens including: `alzheimer`, `plasma`, `biomarker`, `biomarkers`, `predict`.

Now consider two chunks:

- **Chunk A** (from an Alzheimer's paper): mentions `alzheimer`, `plasma`, `biomarker`, `amyloid`, `pTau217`
- **Chunk B** (from a lung cancer paper): mentions `plasma`, `biomarker`, `biomarkers`, `predict`, `detect`, `egfr`, `nsclc`

Chunk B matches four query tokens. It scores high. It has never heard of Alzheimer's.

After ingesting Alzheimer's papers alongside oncology papers in the BioRAG corpus, Chunk B-type results appeared consistently in the top 10. The false positive rate at rank 5 was unacceptable for a clinical decision-support context.

#### Why "Predict" Makes It Worse

The naive fix is: check whether the query's entity tokens (`alzheimer`) appear in the chunk. If not, penalise it.

This works, but it does not go far enough. In the example above, "predict" is in the query and in Chunk B. If you treat "predict" as a discriminative query token, then Chunk B partially escapes the penalty because it matched a query token — just not the disease-specific one.

Here is the insight that required debugging to discover: **clinical verbs are not discriminative**. "Predict", "detect", "identify", "assess", "evaluate" — these words appear in virtually every biomedical paper regardless of disease area. They carry no signal about *which* disease a paper covers. A lung cancer paper predicts things. A diabetes paper detects things. Using their presence in a chunk as evidence of topical relevance is simply noise.

The first version of the reranker treated all query tokens equally when checking entity recall. A lung cancer paper still landed at rank 5 for an Alzheimer's query. It matched "predict", "detect", and "biomarker", so its entity factor was 0.6 + 0.4 × (3/5) = 0.84. That was still too high.

#### The `_GENERIC_BIO_TERMS` Set

The fix is to define a set of terms that appear in virtually every biomedical paper and exclude them from the discriminative-token calculation:

```python
_GENERIC_BIO_TERMS: frozenset[str] = frozenset({
    # Domain nouns ubiquitous across all disease areas
    "biomarker", "biomarkers", "marker", "markers", "disease", "diseases",
    "patient", "patients", "study", "studies", "treatment", "treatments",
    "expression", "level", "levels", "plasma", "blood", "serum",
    "cell", "cells", "protein", "proteins", "gene", "genes",
    ...
    # Common clinical/scientific verbs used in queries about any disease
    "predict", "predicted", "predictive", "predictor", "predictors",
    "detect", "detected", "detectable", "identify", "identified",
    "assess", "assessed", "assessment", "evaluate", "evaluated",
    ...
    # Common adjectives appearing in clinical queries across all diseases
    "early", "late", "novel", "potential", "effective", "accurate",
    "sensitive", "specific", "elevated", "increased", "decreased",
    ...
})
```

This is approximately 80 terms — a deliberately conservative list. "Biomarker" belongs to generic terms. "Alzheimer" does not.

#### The Discriminative-Token Recall Penalty

With this set defined, the discriminative tokens for a query are:

```python
discriminative = query_tokens - self._GENERIC_BIO_TERMS
```

For the query *"What plasma biomarkers predict Alzheimer's disease?"*, the discriminative tokens are `{"alzheimer"}`. "Plasma", "biomarker", "predict", and "disease" are all in `_GENERIC_BIO_TERMS` and get excluded.

The penalty is then applied based on how many discriminative tokens a chunk matches:

```python
if discriminative:
    matched = discriminative & chunk_token_set
    if not matched:
        entity_factor = 0.15   # 85% penalty — demote hard
    else:
        entity_factor = 0.6 + 0.4 * (len(matched) / len(discriminative))
else:
    entity_factor = 1.0  # fully generic query — no penalty
```

Chunk B (the lung cancer paper) matches zero discriminative tokens. Its `entity_factor` is 0.15. Its adjusted score drops to roughly 15% of what BM25 gave it. It falls out of the top 5.

Chunk A (the Alzheimer's paper) matches `alzheimer`. `entity_factor = 0.6 + 0.4 × (1/1) = 1.0`. Full score.

**Why 0.15 and not 0.0?** A hard zero would permanently exclude any chunk that misses the discriminative tokens, even if the reranker's discriminative set was narrowly defined. 0.15 keeps the chunk reachable in edge cases — for instance, a highly general query where all query tokens end up in `_GENERIC_BIO_TERMS`, causing `discriminative` to be empty and triggering the `entity_factor = 1.0` fallback.

#### The Full Scoring Formula

```python
adjusted_score = (
    r.score            # BM25 base score
    * section_w        # section weight (1.0 – 1.4×)
    * (1 + density * 0.4)  # term density bonus (1.0 – 1.4×)
    * position_bonus   # early-rank bonus (decays as 1 + 0.05/rank)
    * entity_factor    # discriminative-token recall (0.15 – 1.0)
)
```

The multiplicative structure ensures that a chunk must earn its place on multiple dimensions simultaneously. A Results chunk from a directly relevant paper outscores an Abstract chunk from a marginally relevant one, which outscores a Methods chunk from an off-topic paper — even if the off-topic paper had a higher raw BM25 score.

---

## The Two-Stage Debugging Process

The reranker described above was not the first version. Here is how it was arrived at.

**Stage 1:** After ingesting Alzheimer's and oncology papers into the same index, the system was tested with the query *"What plasma biomarkers predict Alzheimer's disease?"*. Rank 5 was occupied by an EGFR-mutant NSCLC paper. The paper had never mentioned Alzheimer's. BM25 score: 7.30. Relevance as displayed to the user after normalisation: 75%.

The first fix was to add a check for query entity tokens (capitalised terms extracted by the QueryAnalyzer). If a chunk did not contain "Alzheimer", penalise it. Entity factor for the NSCLC paper: 0.15. Problem solved for this query.

**Stage 2:** Testing a broader query: *"What biomarkers predict early-stage cancer?"*. No specific disease entity in the query. The discriminative set was `{"early-stage", "cancer"}`. A colorectal cancer paper matched "cancer" but not "early-stage". `entity_factor = 0.6 + 0.4 × (1/2) = 0.8`. Reasonable.

But a different failure mode appeared: a diabetes paper that used "predict" and "biomarker" heavily was still appearing at rank 4 for the Alzheimer's query, even after Stage 1 fixes. The discriminative set was `{"alzheimer"}`. The diabetes paper did not contain "alzheimer". So why was its entity_factor not 0.15?

Investigation revealed the bug: "predict" was being added to `discriminative` because it was not yet in `_GENERIC_BIO_TERMS`. The diabetes paper matched "predict" as a discriminative token, so `entity_factor = 0.6 + 0.4 × (1/2) = 0.8`. Still high enough to survive.

The fix: add "predict", "detect", "identify", "assess", and all their morphological variants to `_GENERIC_BIO_TERMS`. After this addition, the diabetes paper's discriminative match dropped to zero. `entity_factor = 0.15`. It left the top 5.

**The lesson:** When an off-topic paper survives an entity penalty at higher-than-expected strength, check whether it matched a query token that *should* have been classified as generic. The diagnostic question is: *would a paper about a completely different disease also contain this token?* If yes, it belongs in `_GENERIC_BIO_TERMS`.

---

## Stage 3: Evidence Classification

After reranking, the pipeline has 5 chunks it believes are relevant. The `EvidenceClassifier` now asks a different question: *in what way is each chunk relevant?*

Three categories:

- **`direct`** — the chunk contains statistical results, significance statements, or explicit confirmations that directly address the query
- **`indirect`** — the chunk is topically related but does not directly answer the question
- **`contradictory`** — the chunk contains language suggesting refutation, failure, or conflicting findings

### Why a Three-Way Split Matters More Than a Continuous Score

A relevance score tells you *how much* a chunk matched the query. Evidence type tells you *how it matched*.

Consider two chunks for the query "Does pTau217 predict Alzheimer's disease?":

- Chunk A: *"pTau217 demonstrated an AUC of 0.94 (95% CI: 0.91–0.97) for discriminating Alzheimer's disease from controls, outperforming all other tested plasma biomarkers."* — **direct**
- Chunk B: *"While pTau217 showed promise in controlled cohorts, results failed to replicate in community-based samples with mixed pathologies."* — **contradictory**

Both chunks are highly relevant. Both match the query entities. But they point in opposite directions. A system that returns only a relevance score conflates them. A clinician who receives only Chunk A has an incomplete picture. The three-way classification surfaces this explicitly in the `DecisionOutput` and flags it as a knowledge gap.

### How Classification Works

The `EvidenceClassifier` applies two ordered pattern lists to the chunk text:

```python
DIRECT_PATTERNS = [
    r'\b(demonstrate|show|reveal|confirm|establish|prove|indicate|suggest)\b',
    r'\b(significant|p\s*[<=>]\s*0\.\d+|95%\s*CI|odds ratio|hazard ratio)\b',
    r'\b(result|finding|observation|data|evidence)\b.*\b(support|consistent)\b',
]

CONTRADICTORY_PATTERNS = [
    r'\b(contradict|inconsistent|conflict|contrary|dispute|challenge|refute)\b',
    r'\b(however|nevertheless|in contrast|on the other hand)\b.*\b(no|not|fail)\b',
    r'\b(no significant|non-significant|failed to|did not|could not)\b',
]
```

Contradictory is checked first. A chunk that matches a contradictory pattern is classified as `contradictory` regardless of whether it also matches a direct pattern. A chunk can simultaneously describe a positive finding and then qualify or refute it. Labelling it `direct` when it contains an explicit contradiction would bury the caveat.

If no contradictory pattern matches, direct patterns are tested. A match on `p < 0.001` or `hazard ratio` is nearly always a quantitative result and the chunk is classified `direct`. If neither pattern family matches, the chunk defaults to `indirect`.

**What the patterns look for:**

The direct patterns are structured around three evidence signatures:

1. **Reporting verbs**: "demonstrate", "reveal", "confirm". These are the verbs authors use when stating a result, as opposed to speculative verbs ("suggest", "propose") used in Discussion sections.
2. **Statistical notation**: `p < 0.05`, `95% CI`, `odds ratio`. The presence of any statistical reporting is strong evidence the chunk contains a primary finding.
3. **Consistency statements**: "evidence consistent with", "findings support". These appear in meta-analyses and systematic reviews when summarising multi-study conclusions.

The contradictory patterns target the linguistic markers of scientific refutation: adversative conjunctions followed by negation ("however ... no"), failure statements ("failed to replicate", "did not reach significance"), and explicit conflict language.

### Key Terms Extraction

Alongside the support type, the classifier extracts `key_terms` — the terms that justify the classification and will surface in the final `DecisionOutput`:

```python
def _extract_key_terms(self, text: str, query_analysis: dict) -> list[str]:
    entities = re.findall(
        r'\b[A-Z][A-Za-z0-9\-]{2,}(?:\s+[A-Z][A-Za-z0-9\-]{2,}){0,2}\b', text
    )
    stats = re.findall(
        r'(?:p\s*[<=>]\s*0\.\d+|OR\s*=?\s*[\d.]+|HR\s*=?\s*[\d.]+|'
        r'95%\s*CI|RR\s*=?\s*[\d.]+|\d+\.?\d*%)', text
    )
    query_terms = [t for t in query_analysis["tokens"] if t in text.lower() and len(t) > 3]
    combined = list(dict.fromkeys(entities[:4] + stats[:2] + query_terms[:3]))
    return combined[:6]
```

Three extraction passes:
- **Named entities** (capitalised multi-word terms): gene names, drug names, disease names
- **Statistical values**: AUC percentages, p-values, OR/HR notation
- **Query-matching terms**: tokens from the original query that appear in the chunk, as confirmation of topical match

The result is a list of up to 6 terms that link the chunk's classification to the source text. This gives the synthesiser and the end user concrete anchors when reading the evidence.

### Limitations and How to Improve

**The patterns are surface-level.** A chunk that uses "suggest" is labelled `direct` because "suggest" appears in `DIRECT_PATTERNS`. But authors use "suggest" both to report their own results ("our data suggest X") and to speculate about mechanisms ("this may suggest X"). The former is direct evidence; the latter is not. Distinguishing them requires parsing sentence structure, not just matching a verb.

**Hedging is not captured.** "pTau217 may be a useful biomarker" and "pTau217 is a validated biomarker" both match direct patterns. The modal "may" qualifies the strength of the claim substantially. A classifier that cannot detect epistemic hedging will overstate the strength of indirect evidence.

**Contradiction requires co-occurrence in one chunk.** A paper can contradict another paper without any single chunk containing contradictory language. The current `EvidenceClassifier` operates at the chunk level; cross-chunk contradiction detection is a job for the `KnowledgeGapDetector` (covered in Post 3).

**How to improve it:** A lightweight BERT-based fact verification model — NLI: does this chunk *entail*, *contradict*, or remain *neutral* relative to the query — would replace all three pattern families with a single inference pass. Models like `MedNLI` or `BioASQ-entailment` are trained exactly for this task on biomedical text. The pattern approach is a zero-dependency baseline that is interpretable and auditable; the NLI approach is more accurate but adds a model dependency.

---

## What the Pipeline Produces After Stage 3

After `Reranker` and `EvidenceClassifier`, the pipeline has transformed 15 raw BM25 candidates into 5 structured `EvidenceNode` objects:

```python
evidence_nodes = [
    EvidenceNode(
        doc_title="Plasma Aβ42/p-Tau217 ratio as a blood-based biomarker...",
        section="Results",
        excerpt="Plasma pTau217 demonstrated an AUC of 0.94 (95% CI 0.91–0.97)...",
        relevance_score=1.000,   # normalized to top result
        support_type="direct",
        key_terms=["pTau217", "AUC", "0.94", "95% CI", "Alzheimer"],
    ),
    EvidenceNode(
        doc_title="Diagnostic Blood-Based Biomarkers for Alzheimer's Disease",
        section="Results",
        excerpt="Amyloid-β42/40 ratio and NfL showed significant associations...",
        relevance_score=0.832,
        support_type="direct",
        key_terms=["Amyloid", "NfL", "significant", "Alzheimer"],
    ),
    EvidenceNode(
        doc_title="Longitudinal Changes in Plasma Biomarkers...",
        section="Discussion",
        excerpt="However, plasma biomarker sensitivity failed to reach clinical...",
        relevance_score=0.641,
        support_type="contradictory",
        key_terms=["plasma", "sensitivity", "failed", "clinical threshold"],
    ),
    # ...
]
```

The lung cancer paper is gone. The Alzheimer's papers are ranked by a combination of BM25 score, section relevance to the diagnosis intent, and entity specificity. The contradictory evidence is flagged explicitly rather than buried.

These nodes are what the `KnowledgeGapDetector` and `AnswerSynthesizer` receive as input.

---

## Where Semantic Search Would Help

Two parts of the reranker are still rule-based in ways that limit them.

Section detection (`_detect_section`) matches literal header strings, so a chunk labelled "Key Findings" or "Clinical Outcomes" falls back to `Body` at weight 1.0 and never receives the Results boost. Embedding the header and comparing it to canonical section descriptions would fix this with no changes to the downstream scoring.

The `_GENERIC_BIO_TERMS` frozenset has the same structural weakness: it is hand-curated, misses morphological variants ("predictability" is not in the list even though "predict" is), and cannot adapt to a corpus that is heavy in one sub-domain where additional terms become universally generic. Replacing it with a corpus-driven genericity score — computed by clustering documents into disease-area groups and measuring how uniformly each vocabulary term appears across clusters — would make the discriminative-token boundary data-derived and self-updating as the corpus grows.

---

## Coming Up in Post 3

Post 3 covers the `KnowledgeGapDetector` and `AnswerSynthesizer`. We will look at how the system decides when it does *not* know the answer and how it constructs an explicit, step-by-step reasoning chain that makes its uncertainty auditable.

The confidence scoring formula derives a 0–1 score from four inputs: evidence count, direct-to-indirect ratio, gap count, and contradiction presence. We will trace why this produces calibrated confidence for sparse evidence and why "calibrated" matters more than "high" in a clinical context.

We will also look at the `FollowUpGenerator` and how intent-aware follow-up questions can drive a user from an initial query toward a more precisely answerable one.

---

*BioRAG is built in pure Python stdlib. The core engine — BM25 index, reranker, evidence classifier, synthesizer — has zero external dependencies. PubMed ingestion requires only `requests`. LLM answer synthesis requires only `anthropic`. Github: https://github.com/ricchasethi/rag-biomedical-decision*
