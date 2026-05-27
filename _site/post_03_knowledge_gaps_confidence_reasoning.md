# Building BioRAG: A Decision-Support RAG System for Biomedical Literature

**Part 3 of 5 — Knowledge Gap Detection, Confidence Scoring, and Reasoning Chains**

---

## Where We Left Off

Post 2 ended with 5 structured `EvidenceNode` objects — promoted from 15 raw BM25 candidates by the Reranker and labelled `direct`, `indirect`, or `contradictory` by the EvidenceClassifier. The lung cancer paper is gone. Contradictory evidence is flagged explicitly. The pipeline knows *what* it found.

The next question is harder: *what didn't it find, and how certain should it be about what it did find?*

These two questions have to be answered before any prose is written. In a clinical decision-support context, the confidence estimate is the system communicating something auditable to a downstream human: *this answer rests on solid direct evidence from multiple corroborating sources* versus *this answer is assembled from partially relevant documents and you should verify it*.

This post covers the three components that answer those questions:

1. `KnowledgeGapDetector` — five structured checks for what the corpus is missing
2. `AnswerSynthesizer` — a four-step reasoning chain and a confidence formula with explicit inputs
3. `FollowUpGenerator` — intent-aware questions that move the user toward a more answerable query

---

## The Full Pipeline After Post 2

Before going deeper, here is what the pipeline has produced at the end of Post 2:

```python
evidence_nodes = [
    EvidenceNode(
        doc_title="Plasma Aβ42/p-Tau217 ratio as a blood-based biomarker...",
        section="Results",
        relevance_score=1.000,
        support_type="direct",
        key_terms=["pTau217", "AUC", "0.94", "95% CI", "Alzheimer"],
    ),
    EvidenceNode(
        doc_title="Diagnostic Blood-Based Biomarkers for Alzheimer's Disease",
        section="Results",
        relevance_score=0.832,
        support_type="direct",
        key_terms=["Amyloid", "NfL", "significant", "Alzheimer"],
    ),
    EvidenceNode(
        doc_title="Longitudinal Changes in Plasma Biomarkers...",
        section="Discussion",
        relevance_score=0.641,
        support_type="contradictory",
        key_terms=["plasma", "sensitivity", "failed", "clinical threshold"],
    ),
    # ...
]
```

These nodes are the input to everything that follows. Two components next are: the `KnowledgeGapDetector` and `AnswerSynthesizer`. Gaps are detected first because they feed into the confidence formula.

---

## Stage 4: Knowledge Gap Detection

The `KnowledgeGapDetector` asks a structurally different question: *what does the returned evidence fail to cover that the query asked about?*

This is the difference between retrieval quality and answer quality. A system can retrieve highly relevant documents and still be unable to answer the specific question. This could be due to the fact that the documents cover the right disease but the wrong endpoint, or the right biomarker but only in a different patient population, or the right intervention but without quantitative effect sizes.

BioRAG implements five gap checks.

### Gap 1: Entity Coverage

Every entity extracted by the `QueryAnalyzer` should appear somewhere in the returned evidence. If a named entity from the query is absent from all five evidence excerpts, that is a coverage failure.

```python
covered_text = ' '.join(e.excerpt for e in evidence).lower()
for entity in query_analysis["entities"]:
    if entity.lower() not in covered_text:
        gaps.append(
            f"No direct evidence found for '{entity}' in the indexed documents"
        )
```

This check is simple and catches the most straightforward failure mode: the corpus was queried for entity X, and entity X does not appear in any of the top-5 evidence nodes. This can happen even when BM25 retrieves documents that mention X, if those documents were demoted by the reranker (perhaps because their section weighting was weak) and replaced by documents that matched generic query tokens without containing the entity.

**What it catches:** A query for "pTau217 and Alzheimer's disease" where the corpus only contains Alzheimer's papers that discuss amyloid-β exclusively — the entity "pTau217" is absent from all returned excerpts.

**What it misses:** Near-synonym failure — a query using "neurofilament light" when the corpus uses "NfL" everywhere, and the abbreviation expansion map does not cover it.

### Gap 2: Quantitative Data

For treatment, prognosis, and epidemiology queries, an answer without numerical evidence is not an answer. It is background knowledge only. The `QueryAnalyzer` flags queries with `needs_quantitative = True` for these intents. The gap detector checks whether at least one evidence node contains statistical notation.

```python
if query_analysis["needs_quantitative"]:
    has_stats = any(
        re.search(r'\d+\.?\d*%|\bp\s*[<=>]|odds ratio|hazard ratio', e.excerpt.lower())
        for e in evidence
    )
    if not has_stats:
        gaps.append(
            "Quantitative data (effect sizes, p-values, confidence intervals) "
            "not found in retrieved evidence"
        )
```

**What this buys you:** For a prognosis query like "What is the 5-year survival rate for stage III NSCLC with osimertinib?", a gap fires if the retrieved evidence contains only mechanism discussions without a single survival figure. The gap propagates into the confidence score (described below) and into the final `DecisionOutput`, where it is visible to the user.

**A common false positive:** A paper that reports percentage values in a non-statistical context, for example, "57% of patients were female" matches `\d+\.?\d*%` and suppresses the quantitative gap. This is tolerable: a paper that reports population percentages is also likely to contain outcome statistics elsewhere in its text.

### Gap 3: Contradictory Evidence

If any evidence node is classified `contradictory`, that contradiction is surfaced as an explicit gap. This is not a retrieval failure — it is a signal about the state of the literature.

```python
contradictions = [e for e in evidence if e.support_type == "contradictory"]
if contradictions:
    gaps.append(
        f"Contradictory evidence present in {len(contradictions)} source(s) — "
        "answer may reflect incomplete consensus"
    )
```

**Why this matters for clinical use:** A system that silently synthesises an answer from three direct-evidence nodes and one contradictory node is a liability. A clinician who receives a confident answer has no reason to suspect that one of the four source papers reports null findings. The contradiction gap ensures this is always visible. It also directly reduces the confidence score, which is described in detail below.

### Gap 4: Low Relevance Scores

BM25 retrieves top-K chunks regardless of whether any of them are actually relevant. If no document in the corpus addresses the query, BM25 still returns 15 results and they are just the best available match from an irrelevant corpus.

The gap detector catches this by checking whether the top raw BM25 result scores above a corpus-size-adjusted threshold. The threshold is derived from the IDF of a rare term — a term appearing in only one document out of N gets an IDF of approximately `log((N + 0.5) / 1.5 + 1)`. If the top result's BM25 score is below `1.5 × idf_rare`, not even a single rare term has matched strongly, which means the corpus very likely does not contain the answer.

```python
if all_results:
    n = max(corpus_chunk_count, 1)
    idf_rare = math.log((n + 0.5) / 1.5 + 1)
    low_score_threshold = idf_rare * 1.5
    if all_results[0].score < low_score_threshold:
        gaps.append(
            "Retrieved evidence has low relevance scores — "
            "the documents may not directly address this query"
        )
```

This threshold scales correctly as the corpus grows. A small corpus of 50 chunks has a lower IDF ceiling than a corpus of 5,000 chunks. The threshold rises as the index grows and a low-quality match becomes harder to hide behind corpus-size effects.

**The practical case this catches:** A user ingests papers on cardiovascular disease and then asks about CRISPR-based cancer therapy. BM25 retrieves the best available CVD papers. Their scores are real scores — not zero — but they reflect only generic term overlap. The low-relevance gap fires, and the `DecisionOutput` tells the user to ingest topic-relevant papers before expecting a meaningful answer.

### Gap 5: Recency

Biomedical evidence ages. A system that confidently synthesises an answer from 2003 papers about biomarker performance is missing a decade of follow-up validation and updated clinical guidelines.

The recency check is a heuristic: if at least half of the evidence nodes contain year-patterns matching the 1990s or early 2000s, a recency gap is flagged.

```python
old_docs = [e for e in evidence if re.search(r'\b(19|200[0-5])\d{2}\b', e.excerpt)]
if len(old_docs) >= len(evidence) // 2 and evidence:
    gaps.append(
        "Evidence may be from older literature — consider verifying with recent publications"
    )
```

This is deliberately coarse. The correct long-term fix is to store the publication year as structured metadata on each document and compare it against the current date. Using regex on excerpt text means year-pattern matches can come from dates in study timelines rather than publication years. But for a zero-dependency core engine, a heuristic is preferable to introducing a structured metadata requirement.

### The Gap List as a First-Class Output

After running all five checks, `KnowledgeGapDetector.detect_gaps()` returns a list capped at five gaps. This list is a first-class field in `DecisionOutput`, not a footnote. It feeds into the confidence formula and is displayed to the user alongside the answer.

```python
gaps = self.gap_detector.detect_gaps(
    q_analysis, evidence_nodes, raw_results, self.index.doc_count
)
# gaps might be:
# [
#   "Contradictory evidence present in 1 source(s) — answer may reflect incomplete consensus",
#   "Quantitative data not found in retrieved evidence",
# ]
```

Both gaps generated here will reduce the final confidence score, as described next.

---

## Stage 5: The Reasoning Chain and Confidence Scoring

`AnswerSynthesizer.synthesize()` has two jobs: build an explicit reasoning chain and compute a confidence score. These are connected — the reasoning chain documents the inputs that feed into the confidence formula, making the score interpretable rather than opaque.

### The Four-Step Reasoning Chain

Every query produces a `reasoning_chain` — a list of `ReasoningStep` objects, each with a label, content string, and a step-level confidence score. The structure is fixed: steps 1 and 2 always execute; steps 3 and 4 only execute if there is evidence and gaps, respectively.

```python
@dataclass
class ReasoningStep:
    step_number: int
    label: str
    content: str
    confidence: float
```

**Step 1 — Query interpretation (confidence: 1.0)**

The query analyzer's output is summarised: intent classification, named entities found, number of expanded tokens used in retrieval. Confidence is always 1.0 at this step because the query was parsed without information loss — even if the parse was incorrect, the parsing itself succeeded.

```
Query classified as 'diagnosis' intent.
Key entities identified: Alzheimer.
Search performed with 8 expanded tokens.
```

This step gives the user something to verify: if the intent was misclassified, they can see it here and formulate a corrected query.

**Step 2 — Evidence assessment (confidence: variable)**

The evidence count breakdown — direct, indirect, contradictory — is computed and its step-level confidence is calculated from the ratio of strong evidence to total:

```python
confidence = min(1.0,
    (direct_count * 0.3 + indirect_count * 0.15) / max(len(evidence), 1) + 0.4
)
```

At this step, the formula represents how strong the evidence pool is in isolation, before gaps are taken into account. Five direct evidence nodes yield approximately `(5 × 0.3) / 5 + 0.4 = 0.70`. Three direct and two indirect yield `(3 × 0.3 + 2 × 0.15) / 5 + 0.4 = 0.64`. The base of 0.4 ensures that even a weak evidence set starts with a non-zero assessment rather than collapsing to zero at this step.

```
Retrieved 5 evidence node(s): 2 direct, 2 indirect, 1 contradictory.
Sources span 4 document(s).
```

**Step 3 — Answer synthesis (confidence: variable)**

This step is set after the answer text is generated. Its confidence comes from `_compute_synthesis_confidence()`:

```python
def _compute_synthesis_confidence(self, evidence: list[EvidenceNode]) -> float:
    avg_relevance = sum(e.relevance_score for e in evidence) / len(evidence)
    direct_ratio = sum(1 for e in evidence if e.support_type == "direct") / len(evidence)
    contra_penalty = sum(1 for e in evidence if e.support_type == "contradictory") * 0.1
    return max(0.0, min(1.0, avg_relevance * 0.5 + direct_ratio * 0.5 - contra_penalty))
```

Three signals here. `avg_relevance` is the mean of the normalized relevance scores across the five evidence nodes. It reflects how well the reranker scores the documents after all three penalty factors are applied. `direct_ratio` is the fraction of evidence that contains explicit statistical results or confirmatory language. Each contradictory evidence node subtracts a flat 0.1 penalty.

This formula is what actually penalises contradictions at the synthesis step. A set of five evidence nodes where three are direct, one is indirect, and one is contradictory, with average relevance of 0.85, yields: `0.85 × 0.5 + 0.6 × 0.5 − 0.1 = 0.625`. Replace the contradictory node with another direct node and it rises to `0.85 × 0.5 + 0.8 × 0.5 = 0.825`.

**Step 4 — Knowledge gap analysis (confidence: 0.7)**

If gaps were detected, they are summarised at this step. The step confidence is fixed at 0.7 — the system is fairly confident in its gap identification but acknowledges that the gap detector is heuristic and may miss gaps it could not pattern-match.

```
Identified 2 gap(s): Contradictory evidence present in 1 source(s) — answer may
reflect incomplete consensus; Quantitative data not found in retrieved evidence.
```

### The Overall Confidence Formula

After the reasoning chain is assembled, the final confidence score is computed by `_compute_overall_confidence()`. This is the number that surfaces in `DecisionOutput.confidence` and drives the label (`"High"`, `"Moderate"`, `"Low"`, `"Insufficient"`).

```python
def _compute_overall_confidence(
    self,
    evidence: list[EvidenceNode],
    gaps: list[str],
    direct_count: int,
) -> float:
    base = 0.3
    evidence_bonus = min(0.4, len(evidence) * 0.08)
    direct_bonus = min(0.2, direct_count * 0.07)
    gap_penalty = len(gaps) * 0.05
    return max(0.05, min(0.95, base + evidence_bonus + direct_bonus - gap_penalty))
```

Four inputs, each with a bounded contribution:

| Input | Effect | Range |
|-------|--------|-------|
| Evidence count | +0.08 per node, capped at +0.40 | 0 to 0.40 |
| Direct evidence count | +0.07 per direct node, capped at +0.20 | 0 to 0.20 |
| Gap count | −0.05 per gap | uncapped downward |
| Base | always 0.30 | — |

The maximum achievable score is 0.95 (`max(0.05, min(0.95, ...))`). The formula never returns 0.0 or 1.0, which is intentional. Absolute zero confidence would mean the system refuses to respond at all; absolute certainty is epistemically dishonest in biomedical literature synthesis. The ceiling of 0.95 is a statement that a probabilistic system over noisy biomedical text should never claim to be certain.

**Worked examples:**

*Strong evidence case:* 5 evidence nodes, 4 direct, 0 gaps.
`0.3 + min(0.4, 5 × 0.08) + min(0.2, 4 × 0.07) − 0 = 0.3 + 0.4 + 0.2 = 0.90 → "High"`

*Weak evidence case:* 3 evidence nodes, 1 direct, 3 gaps (contradiction, no stats, low relevance).
`0.3 + min(0.4, 3 × 0.08) + min(0.2, 1 × 0.07) − (3 × 0.05) = 0.3 + 0.24 + 0.07 − 0.15 = 0.46 → "Moderate"`

*Corpus miss case:* 2 evidence nodes, 0 direct, 4 gaps (entity missing, low relevance, no stats, recency).
`0.3 + min(0.4, 2 × 0.08) + 0 − (4 × 0.05) = 0.3 + 0.16 − 0.2 = 0.26 → "Low"`

### Why Calibrated Beats High

The temptation in any retrieval system is to make confidence scores look high. High-confidence answers feel more useful. But in a clinical context, a miscalibrated high-confidence answer that rests on weak evidence is more dangerous than a correctly low-confidence answer that tells the user to seek additional evidence.

BioRAG's formula is deliberately conservative. The base of 0.3 means an answer from zero direct evidence and zero gaps starts at 30% — `"Low"`. You need 4+ evidence nodes with 2+ direct classifications to breach `"High"`. This is not a design flaw; it is a design choice: the system should default to skepticism and require evidence to move the needle upward.

The consequence is that `"Moderate"` is the most common label in practice, which is correct for a system answering biomedical research questions from an indexed corpus that is always narrower than the published literature.

---

## The Answer Synthesizer: Building the Prose

After confidence is computed, `_build_answer()` assembles the actual response text from the evidence nodes. The rule-based synthesizer — the core engine default — constructs the answer in three passes.

**Pass 1: Primary evidence summary**

Direct evidence nodes are listed first, up to three, with each excerpt truncated at a sentence boundary to stay under 300 characters. If no direct evidence exists, the three best indirect nodes are used instead, with a framing caveat. If neither exists, the answer states that evidence is insufficient.

The sentence-boundary truncation is worth emphasising. Cutting at 300 raw characters would routinely end mid-sentence, creating excerpts that are syntactically incomplete and misleading. `TextProcessor.truncate_at_sentence()` scans forward from 300 characters to the next sentence boundary, ensuring that every displayed excerpt is a complete claim.

**Pass 2: Contradiction warning**

If any contradictory nodes exist, they are surfaced separately with an explicit warning marker. In the rule-based synthesizer this is the `⚠` prefix. In the LLM synthesizer (described next), the model is instructed to acknowledge the contradiction in-line with a citation.

**Pass 3: Cross-source convergence**

If two or more evidence nodes share key terms like biomarker names, gene identifiers, statistical values, then `_find_common_terms()` surfaces those as a convergence signal:

```python
def _find_common_terms(self, evidence: list[EvidenceNode]) -> list[str]:
    term_counts: Counter = Counter()
    for e in evidence:
        term_counts.update(set(e.key_terms))
    return [term for term, count in term_counts.most_common(6) if count >= 2]
```

A term appearing in three out of five evidence nodes is a signal worth surfacing. It is not proof of consensus as two papers can cite the same term in opposite conclusions, but repeated mention across independent sources is at minimum a flag that this term is centrally relevant to the question.

---

## The LLM Answer Synthesizer

The rule-based `_build_answer()` is adequate for surfacing evidence, but it is structured text assembly. For users who want a natural language answer, BioRAG provides `ClaudeSynthesizer`, a drop-in replacement that swaps only `_build_answer()` for a Claude API call.

Critically: every other part of the pipeline is identical. The reasoning chain construction, confidence scoring, gap detection, and evidence classification all run through the same code path. The LLM only touches the final prose generation step.

```python
engine = BioRAGEngine(synthesizer=ClaudeSynthesizer())
result = engine.query("What plasma biomarkers predict Alzheimer's disease?")
```

### The Grounding Contract

The system prompt enforces a strict grounding contract:

```
Every factual claim in your answer MUST be traceable to at least one of
the supplied excerpts. Cite sources inline as [N] where N is the excerpt number.
Do NOT use knowledge from your training data. If the excerpts do not contain
sufficient information, say so explicitly.
```

This is the core safety property of the LLM synthesizer. A model that draws on training knowledge to fill gaps in the retrieved evidence is a hallucination risk in exactly the contexts where hallucinations are most harmful. The system prompt forbids it explicitly and instructs the model to surface evidence insufficiency as a failure mode rather than paper over it.

### Evidence Formatting in the User Message

The user message passes each evidence node as a numbered excerpt with metadata:

```
[1] Plasma Aβ42/p-Tau217 ratio as a blood-based biomarker...
    | Section: Results | Relevance: 100% | Type: DIRECT
Excerpt: Plasma pTau217 demonstrated an AUC of 0.94 (95% CI 0.91–0.97)...

[2] Diagnostic Blood-Based Biomarkers for Alzheimer's Disease
    | Section: Results | Relevance: 83% | Type: DIRECT
Excerpt: Amyloid-β42/40 ratio and NfL showed significant associations...

[3] Longitudinal Changes in Plasma Biomarkers...
    | Section: Discussion | Relevance: 64% | Type: CONTRADICTORY
Excerpt: However, plasma biomarker sensitivity failed to reach clinical...
```

The `DIRECT` / `INDIRECT` / `CONTRADICTORY` type label is included in the excerpt header so Claude knows how to weight each source before generating prose. A CONTRADICTORY source should be acknowledged differently than a DIRECT one.

### Prompt Caching

The system prompt, which is long and static, is cached with `"cache_control": {"type": "ephemeral"}`. This reduces API latency and cost for repeated queries in the same session because the system prompt tokens are only processed once.

```python
system=[
    {
        "type": "text",
        "text": SYSTEM_PROMPT,
        "cache_control": {"type": "ephemeral"},
    }
]
```

For a decision-support session where a clinician is iterating through a series of related queries, prompt caching cuts the effective token cost of the system prompt to near zero after the first query in a session.

### Fallback on API Failure

If the Claude API call fails for any reason — network error, rate limit, model unavailability — `ClaudeSynthesizer` falls back to the parent class's rule-based `_build_answer()` and appends a note to the reasoning chain indicating fallback occurred. The confidence score and gap list are unaffected.

```python
except Exception as exc:
    answer_parts = self._build_answer(query_analysis, evidence)
    answer_text = "\n\n".join(answer_parts)
    synthesis_note = f"LLM synthesis failed ({exc}); used rule-based fallback."
```

This fallback is a design requirement. A clinical decision-support system that becomes unusable when an external API is unavailable is not suitable for production use.

---

## Stage 6: The Follow-Up Generator

After the answer is synthesised, `FollowUpGenerator.generate()` produces up to four follow-up questions. The goal is not to prompt the user with generic next questions, but, to move them from a query that the system answered imperfectly toward a query that is more precisely answerable.

### Intent-Aware Templates

The follow-up templates are keyed to the query's intent classification. A `diagnosis` query and a `treatment` query need very different follow-ups:

```python
TEMPLATES = {
    "diagnosis": [
        "What is the sensitivity and specificity of this diagnostic approach?",
        "What are the differential diagnoses to consider?",
        "Are there validated clinical scoring systems for this condition?",
    ],
    "treatment": [
        "What is the recommended dosage range and administration route?",
        "What are the most common adverse effects reported?",
        "Is there evidence of treatment resistance or tolerance?",
    ],
    "mechanism": [
        "What are the downstream effects of {entity}?",
        "Which molecular targets are involved in this pathway?",
        "Are there known inhibitors or activators of this mechanism?",
    ],
    # ...
}
```

The `{entity}` placeholder is filled with the first entity extracted by the `QueryAnalyzer`. For a query about "BRCA1 mutation mechanisms", the first follow-up becomes "What are the downstream effects of BRCA1?" — a question that narrows from general mechanism to consequence.

### Gap-Driven Questions

The fourth follow-up question is generated from the key terms shared across the top two evidence nodes:

```python
key_terms = []
for e in evidence[:2]:
    key_terms.extend(e.key_terms[:2])
unique_terms = list(dict.fromkeys(key_terms))[:2]
questions.append(
    f"What is the clinical significance of {' and '.join(unique_terms)}?"
)
```

This is the follow-up generator's most practical contribution. If the system retrieved evidence about "pTau217" and "amyloid-β42/40", the follow-up is "What is the clinical significance of pTau217 and amyloid-β42/40?". This is gap-driven in the sense that it surfaces the entities the system found most relevant, inviting the user to go deeper on those specific signals.

---

## What the Full Pipeline Produces

At the end of all six stages, `BioRAGEngine.query()` returns a `DecisionOutput`:

```python
@dataclass
class DecisionOutput:
    query: str
    answer: str
    confidence: float
    confidence_label: str          # "High" | "Moderate" | "Low" | "Insufficient"
    evidence: list[EvidenceNode]
    reasoning_chain: list[ReasoningStep]
    knowledge_gaps: list[str]
    follow_up_questions: list[str]
    sources_used: int
    total_chunks_searched: int
```

For the query "What plasma biomarkers predict Alzheimer's disease?", a well-indexed corpus produces something like this:

```python
DecisionOutput(
    query="What plasma biomarkers predict Alzheimer's disease?",
    confidence=0.81,
    confidence_label="High",
    sources_used=4,
    total_chunks_searched=312,

    evidence=[
        EvidenceNode(doc_title="Plasma Aβ42/p-Tau217 ratio...",
                     section="Results", relevance_score=1.000,
                     support_type="direct",
                     key_terms=["pTau217", "AUC", "0.94", "95% CI", "Alzheimer"]),
        EvidenceNode(doc_title="Diagnostic Blood-Based Biomarkers...",
                     section="Results", relevance_score=0.832,
                     support_type="direct",
                     key_terms=["Amyloid", "NfL", "significant", "Alzheimer"]),
        EvidenceNode(doc_title="Longitudinal Changes in Plasma Biomarkers...",
                     section="Discussion", relevance_score=0.641,
                     support_type="contradictory",
                     key_terms=["plasma", "sensitivity", "failed", "clinical threshold"]),
    ],

    reasoning_chain=[
        ReasoningStep(step_number=1, label="Query interpretation",
                      content="Query classified as 'diagnosis' intent. "
                              "Key entities: Alzheimer. "
                              "Search performed with 8 expanded tokens.",
                      confidence=1.0),
        ReasoningStep(step_number=2, label="Evidence assessment",
                      content="Retrieved 5 node(s): 2 direct, 2 indirect, 1 contradictory. "
                              "Sources span 4 document(s).",
                      confidence=0.64),
        ReasoningStep(step_number=3, label="Answer synthesis",
                      content="Synthesized from 5 excerpt(s). "
                              "Primary: 'Plasma Aβ42/p-Tau217 ratio' (Results). "
                              "Note: contradictory evidence in corpus.",
                      confidence=0.68),
        ReasoningStep(step_number=4, label="Knowledge gap analysis",
                      content="Identified 1 gap(s): Contradictory evidence present...",
                      confidence=0.7),
    ],

    knowledge_gaps=[
        "Contradictory evidence present in 1 source(s) — "
        "answer may reflect incomplete consensus"
    ],

    follow_up_questions=[
        "What is the sensitivity and specificity of this diagnostic approach?",
        "What are the differential diagnoses to consider?",
        "Are there validated clinical scoring systems for this condition?",
        "What is the clinical significance of pTau217 and Amyloid?",
    ],
)
```

Every field is auditable. The confidence label "High" is traceable to the formula: `0.3 + 0.4 + 0.14 − 0.05 = 0.79` (5 nodes, 2 direct, 1 gap). The contradictory evidence node is visible in `evidence` and referenced in both `reasoning_chain` step 3 and `knowledge_gaps`. The follow-up questions are grounded in both the query's classified intent and the actual terms found in the evidence.

Nothing is hidden. Nothing is inferred silently.

---

## Limitations and How to Improve

### The Rule-Based Synthesizer Is Not a Prose Writer

`_build_answer()` produces structured text, not analysis. It concatenates excerpts; it does not argue from them. For a researcher who needs a synthesised interpretation — not just a list of relevant passages — the rule-based synthesizer is a starting point, not a destination. The `ClaudeSynthesizer` addresses this, but at the cost of an external API dependency.

**Short-term improvement:** Add a cross-source comparison step to `_build_answer()`. When two direct-evidence nodes report different numerical values for the same endpoint (e.g., one AUC of 0.94 and another of 0.81 for plasma pTau217), note the discrepancy explicitly rather than listing both neutrally.

### Gap Detection Is Heuristic

All five gap checks rely on pattern matching over text. The quantitative gap check catches `\d+\.?\d*%` and misses narratively described results ("a three-fold reduction"). The recency check matches year patterns in excerpt text rather than structured publication metadata. The entity coverage check is case-sensitive and fails when the corpus uses different capitalisation or abbreviation conventions than the query.

**Medium-term improvement:** Ingest structured metadata — publication year, journal, study design — alongside document text, and run gap checks against those structured fields rather than regex on excerpts. This also enables a new gap type: study-design gap (all retrieved evidence is from observational studies when the query asks about RCT evidence).

### Confidence Is Uncalibrated

The formula produces a score between 0.05 and 0.95, but there is no ground truth against which to calibrate it. A confidence of 0.81 should ideally mean the system is correct 81% of the time on queries that produce that score. Without a labelled answer-quality eval set, that claim cannot be verified.

**Long-term improvement:** Build an answer-quality evaluation harness alongside the existing retrieval eval harness (covered in Post 5). For a set of queries with known correct answers, run the full pipeline and measure whether high-confidence answers are more often correct than low-confidence ones. If the curve is not monotonic, the formula coefficients need adjustment. This is calibration in the statistical sense — it is harder to implement than retrieval eval but more important for clinical use.

### `FollowUpGenerator` Is Lookup-Based

The template lookup produces reasonable follow-ups for common intents, but it has no awareness of what was actually missing from the evidence. A gap-driven follow-up that says "consider searching for quantitative effect sizes" is more useful than "what patient populations were studied?" when the quantitative gap fired.

**Short-term improvement:** Use the detected gaps as direct inputs to follow-up generation. If the quantitative gap fired, prepend "What are the reported effect sizes and confidence intervals for this finding?" as the first follow-up, regardless of intent template. If the entity coverage gap fired, generate a follow-up that names the missing entity: "What evidence exists specifically for {missing entity}?"

---

## Coming Up in Post 4

Post 4 covers the full LLM answer synthesis path in depth. We will look at prompt engineering decisions — why the system prompt forbids training knowledge use, what makes the evidence formatting in the user message effective, and how to test that the grounding constraint is actually holding.

We will trace a failure case where the model partially violated the grounding rule despite the instruction — what the output looked like, how it was detected, and what prompt change fixed it.

We will also look at how the full pipeline integrates with the FastAPI server (`server.py`) and the MCP server (`mcp_server.py`), and what it means in practice to call BioRAG as a tool from within a Claude Code session using the registered MCP interface.

---

*BioRAG is built in pure Python stdlib. The core engine — BM25 index, reranker, evidence classifier, synthesizer — has zero external dependencies. PubMed ingestion requires only `requests`. LLM answer synthesis requires only `anthropic`.*
