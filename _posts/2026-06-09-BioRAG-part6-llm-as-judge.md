---
layout: post
title: "BioRAG: Part 6"
series: "BioRAG"
part: 6
date: 2026-06-09
---

# Building BioRAG: A Decision-Support RAG System for Biomedical Literature

**Part 6 — LLM-as-Judge: Why Good Retrieval Does Not Guarantee Good Answers**

---

In Part 5 of this series, I measured retrieval quality using MRR and NDCG. The numbers looked good where 9 out of 10 queries hit MRR = 1.0, meaning the right document landed at rank 1 almost every time.

So I assumed the system was working.

Then I ran a second eval. This one scored the actual *answer text* and the picture changed completely. The average answer quality with the rule-based synthesizer was **5.4 out of 10** on an eight-dimension rubric. The retrieval was nearly perfect. The answers were not.

This post is about that gap and how I used Claude as an automated judge to measure it.

---

## Why Retrieval Metrics Are Not Enough

MRR and NDCG answer one question: *did the right document appear near the top of the ranked list?* They say nothing about what happened to that document after retrieval.

In a RAG pipeline, the answer goes through at least two more stages after retrieval: reranking and synthesis. Both can lose information. The synthesizer might:

- Quote a passage without drawing a directional conclusion from it
- Name a disease area without naming the specific biomarker
- Reproduce introductory sentences when the results section contains the actual answer
- Surface the right document at rank 1 but pick a chunk from the discussion rather than results

None of these failures show up in MRR or NDCG. You need a second eval that looks at the answer text itself.

---

## Designing the Rubric

The rubric for answer quality needs to be specific enough to catch real failures but generic enough to apply across different disease areas and query types.

After testing several framings, I settled on eight dimensions:

**1. Semantic Coverage (0–2)**
Does the answer address the specific biological phenomenon or just the topic area?

- 0 — Only the broad topic is covered (e.g. "biomarkers in Alzheimer's disease")
- 1 — The specific phenomenon is engaged but incomplete
- 2 — The same biological conclusion as the reference claim in ground truth

**2. Entity Coverage (0–2)**
Are the specific genes, proteins, drugs, or markers named and in the right context?

- 0 — None of the expected entities appear
- 1 — Some entities named or all named but in an irrelevant context
- 2 — All key entities named in the correct context

**3. Directional Agreement (0–1)**
Does the answer state the direction of effect explicitly?

- 0 — Direction absent, unclear, or contradicted
- 1 — Direction matches (e.g. "p-tau is elevated", "uric acid is reduced")

**4. Quantitative Detail (0–1)**
Are magnitudes, effect sizes, or statistics consistent with the evidence?

- 0 — No quantitative information, or magnitudes inconsistent
- 1 — At least one magnitude consistent with the reference

**5. Contextual Accuracy (0–1)**
Is the finding placed in the correct specific context?

- 0 — Wrong context, no context, or generic language only
- 1 — The claim-specific context is named explicitly (timepoint, subgroup, tissue)

**6. Source Attribution (0–1)**
Are factual claims linked to inline numbered citations?

- 0 — Assertions made without inline citations, or no [N] references appear
- 1 — Every substantive factual claim has at least one inline citation ([1], [2], etc.)

**7. Evidence Strength (0–1)**
Is the study design or evidence type explicitly named?

- 0 — Findings stated without characterising the evidence (no RCT, meta-analysis, cohort, in vitro, n= mentioned)
- 1 — At least one evidence type or study design explicitly named

**8. Uncertainty Calibration (0–1)**
Does expressed confidence match evidence quality?

- 0 — Uniform assertiveness regardless of evidence strength, or over-hedges strong evidence
- 1 — Strong evidence → clear assertion; weak/preliminary → hedged language ("suggests", "preliminary")

**Maximum: 10 points per query.**

The last three dimensions are particularly important for clinical decision support. Source attribution ensures claims are auditable. Evidence strength tells clinicians how much weight to put on a finding. Uncertainty calibration prevents both false confidence and excessive hedging.

The scoring intentionally rewards explicitness. An answer that *implies* a direction without stating it scores 0 on directional agreement. An answer that names a disease without naming the specific tissue or patient subgroup scores 0 on contextual accuracy.
This strictness reveals the difference between answers that are technically correct and answers that are clinically useful.

---

## Implementing the Judge

The judge is a separate Claude call that receives:

1. The reference claim (what a correct answer should assert)
2. The expected entities, direction, and context
3. The actual answer text produced by the pipeline

I used Claude's `tool_use` API with a strict JSON schema for the rubric dimensions.
This forces integer scores in the declared enum values.

```python
_JUDGE_TOOL = {
    "name": "score_answer",
    "input_schema": {
        "type": "object",
        "properties": {
            "semantic_coverage":      {"type": "integer", "enum": [0, 1, 2]},
            "entity_coverage":        {"type": "integer", "enum": [0, 1, 2]},
            "directional_agreement":  {"type": "integer", "enum": [0, 1]},
            "quantitative_detail":    {"type": "integer", "enum": [0, 1]},
            "contextual_accuracy":    {"type": "integer", "enum": [0, 1]},
            "source_attribution":     {"type": "integer", "enum": [0, 1]},
            "evidence_strength":      {"type": "integer", "enum": [0, 1]},
            "uncertainty_calibration":{"type": "integer", "enum": [0, 1]},
            "rationale":              {"type": "string"},
        },
        "required": [...]
    }
}
```

The judge is deliberately isolated from the retrieval context. It only sees the answer text and the reference claim. This prevents the judge from giving credit for information that was retrieved but not expressed in the answer.

The system prompt instructs the judge to score based strictly on what is explicitly
stated and not implied:

> *Base your scores strictly on what the AI answer explicitly states — do not infer or 
> credit implied knowledge. An answer that implies a direction without stating it
> scores 0 on directional agreement.*

---

## Reference Claims

For each of the 10 eval queries, I wrote a reference claim specifying exactly what
the answer should assert. Here is an example:

**Query:** Does resveratrol supplementation reduce uric acid levels in type 2 diabetes?

**Reference claim:** "Resveratrol supplementation significantly reduces serum uric acid in type 2 diabetes patients, with a pooled weighted mean difference of −0.42 mg/dL (P = 0.0005) and low heterogeneity (I² = 0%) across 12 RCTs with 636 participants."

**Expected entities:** resveratrol, uric acid, WMD, RCT

**Expected direction:** decreased serum uric acid

**Expected context:** type 2 diabetes patients, meta-analysis of RCTs

The reference claims are grounded in the actual corpus documents, so they reflect what is genuinely retrievable.

---

## Results: Rule-Based vs LLM Synthesizer

I ran the eval twice. Once with the default rule-based synthesizer (zero external
dependencies, pure Python), and once with `ClaudeSynthesizer` (which calls
`claude-sonnet-4-6` with the retrieved evidence formatted as numbered excerpts).

### Summary scores

| Dimension | Rule-based | Claude | Δ |
|---|---|---|---|
| Semantic Coverage | 1.20 / 2 | 1.40 / 2 | +0.20 |
| Entity Coverage | 1.10 / 2 | 1.70 / 2 | +0.60 |
| Directional Agreement | 0.40 / 1 | **0.80 / 1** | **+0.40** |
| Quantitative Detail | 0.20 / 1 | 0.30 / 1 | +0.10 |
| Contextual Accuracy | 0.80 / 1 | 1.00 / 1 | +0.20 |
| Source Attribution | 0.90 / 1 | 1.00 / 1 | +0.10 |
| Evidence Strength | 0.50 / 1 | 0.40 / 1 | −0.10 |
| Uncertainty Calibration | 0.30 / 1 | **0.90 / 1** | **+0.60** |
| **Total** | **5.40 / 10** | **7.50 / 10** | **+2.10** |

### Combined retrieval + answer quality table

| Query | MRR | NDCG@3 | Rule-based Tot | Claude Tot |
|---|---|---|---|---|
| Q01 — Blood biomarkers for preclinical AD | 1.00 | 1.00 | 5 | 5 |
| Q02 — Plasma amyloid-beta accuracy | 1.00 | 1.00 | 4 | 6 |
| Q03 — p-tau diagnostic performance | 0.33 | 0.50 | 4 | 6 |
| Q07 — Biomarkers for early AD intervention | 1.00 | 1.00 | 2 | **8** |
| Q08 — Resveratrol and uric acid | 1.00 | 1.00 | **10** | **10** |
| Q09 — Resveratrol and SCr/BUN | 1.00 | 1.00 | **10** | **10** |
| Q11 — PD-L1 in EGFR-mutant NSCLC | 1.00 | 1.00 | 6 | 7 |
| Q12 — TKI vs ICI in NSCLC | 1.00 | 1.00 | 4 | 6 |
| Q13 — Treatment of CRKP | 1.00 | 1.00 | 4 | 7 |
| Q14 — Eravacycline vs KPC-2/NDM-1 | 1.00 | 1.00 | 5 | **10** |

---

## Comparing with RAGAS

After building and running the custom judge, I wanted to see whether a standard framework reaches the same conclusions or not. I used RAGAS, the most widely used open-source RAG evaluation library. I implemented the same eight-dimension rubric using RAGAS.

RAGAS 0.1.21 provides two relevant metric types:

- **`LabelledRubricsScore`** — scores answers against a rubric with 1–3 labeled levels. Used for Semantic Coverage and Entity Coverage (our 0–2 dimensions).
- **`AspectCritique`** — binary yes/no judgment against a natural-language definition. Used for all six binary dimensions: Directional Agreement, Quantitative Detail, Contextual Accuracy, Source Attribution, Evidence Strength, and Uncertainty Calibration.

Both are wired to Claude via LangChain's `ChatAnthropic` wrapper, so the same model
judges both approaches. The critical difference is what context the judge receives.

In the custom implementation, the judge prompt explicitly passes:
- The reference claim
- The list of expected entities
- The expected direction of effect
- The expected specific context

In the RAGAS implementation, the rubric level descriptions are generic natural language i.e. no expected entities, no expected direction. The judge must infer correctness from the rubric description alone.

### Three-way comparison

| Query | Custom/Rule | Custom/LLM | RAGAS/Rule | Δ (RAGAS − Custom) |
|---|---|---|---|---|
| Q01 — Blood biomarkers for preclinical AD | 5 | 5 | 0.0 | −5.0 |
| Q02 — Plasma amyloid-beta accuracy | 4 | 6 | 0.0 | −4.0 |
| Q03 — p-tau diagnostic performance | 4 | 6 | 1.0 | −3.0 |
| Q07 — Biomarkers for early AD intervention | 2 | **8** | 0.0 | −2.0 |
| Q08 — Resveratrol and uric acid | **10** | **10** | 8.0 | −2.0 |
| Q09 — Resveratrol and SCr/BUN | **10** | **10** | 7.0 | −3.0 |
| Q11 — PD-L1 in EGFR-mutant NSCLC | 6 | 7 | 1.0 | −5.0 |
| Q12 — TKI vs ICI in NSCLC | 4 | 6 | 3.0 | −1.0 |
| Q13 — Treatment of CRKP | 4 | 7 | 0.0 | −4.0 |
| Q14 — Eravacycline vs KPC-2/NDM-1 | 5 | **10** | 3.0 | −2.0 |
| **Mean** | **5.40** | **7.50** | **2.30** | **−3.10** |

### Why RAGAS scores significantly lower

RAGAS gives a mean of **2.30/10** versus our custom judge's 5.40/10. The three dimensions (Source Attribution, Evidence Strength, and Uncertainty Calibration) make this divergence even more visible: RAGAS scores 0.00/1 on Source Attribution and Uncertainty Calibration for all but the best-structured documents, because its generic `AspectCritique` definitions lack the domain-specific context to judge these properties accurately. The divergence is largest on the Alzheimer's and infectious disease queries requiring expert knowledge to assess.

**Q01, Q02, Q07, and Q13 score 0/10 under RAGAS** but between 2/10 and 5/10 under our judge. The RAGAS rubric descriptions give the model no indication of what a correct answer looks like for these domain-specific queries. A rubric level that says "the answer generally aligns with the ground truth" cannot tell the judge whether preclinical Alzheimer's detection or KPC-2/NDM-1 co-production were addressed. This means the model lacks sufficient domain context to infer it from generic level descriptions.

**Source Attribution and Uncertainty Calibration score 0.00 across all RAGAS queries.**
These are the two dimensions where the gap is most glaring. Our custom judge prompt explicitly defines what counts as a citation ([N] format) and what counts as calibrated hedging. RAGAS receives these as generic `AspectCritique` definitions with no examples, and the rule-based synthesizer's citation format (which uses [N] consistently) is simply not recognised by the generic judge as satisfying attribution criteria.

**Q08 and Q09 are the closest to three-way agreement.** The resveratrol findings are
so explicit — WMD −0.42, P = 0.0005, 12 RCTs, T2DM — that even a generic judge reaches a similar score. Q08 in RAGAS scores 8/10 (vs 10/10 custom), only missing Source Attribution and Uncertainty Calibration.

**Contextual Accuracy collapses in RAGAS (0.30/1).** Our `AspectCritique` definition
says explicitly: *"Generic phrases like 'in patients' do not qualify."* That strictness is in our prompt. The RAGAS equivalent lacks this instruction, so broad contextual language earns credit it would not receive from a domain-grounded judge.

### What this tells us about judge design

The comparison confirms a practical rule: **the judge is only as good as the context
you give it.** Generic rubric levels produce generic scores. Domain-specific rubrics
that pass expected entities and directions into the judge prompt produce scores that
track clinical utility rather than just surface-level relevance.

RAGAS is the right choice when you need benchmark comparability across systems and
domains. The custom judge is the right choice when you need scores that are grounded
in what your specific corpus actually contains and what your users actually need to know.

---

## What the Results Reveal

### Finding 1: Retrieval is the solved problem; synthesis is where answers fail

9 out of 10 queries hit MRR = 1.0. The right document was retrieved. But the average
answer quality with the rule-based synthesizer was 5.40/10. It is well below what a careful reader of the source paper would produce.

This is the core gap that retrieval metrics hide. High MRR tells you the engine found the evidence. It says nothing about whether that evidence was turned into a useful clinical statement.

### Finding 2: Directional agreement and uncertainty calibration are the biggest differentiators

The rule-based synthesizer scored 0.40/1 on directional agreement and only 0.30/1 on
uncertainty calibration. Claude scored 0.80/1 on both — a full doubling on direction and a tripling on calibration.

Why the gap on uncertainty calibration? The rule-based synthesizer uses a fixed template:
"Based on direct evidence from N source(s), the following can be stated…" It applies
the same assertive framing regardless of whether the evidence is a 12-RCT meta-analysis or a single abstract. Claude reads the evidence and adjusts its language: strong evidence gets clear assertions and incomplete evidence gets explicit hedging and a note that the corpus lacks sufficient data.

For clinical decision support, both properties matter equally. "This drug affects uric acid" is not the same as "this drug reduces uric acid." And "resveratrol reduces uric acid" said with the same confidence as "eravacycline may have activity in vitro, though clinical data are lacking" gives the clinician no signal about how much to rely on either.

### Finding 3: Quantitative detail remains the hardest dimension

Both synthesizers score poorly here — 0.20 and 0.30 respectively. The gap between
them is the smallest of any dimension.

This is a chunking problem and not a synthesis problem. The statistics live in the Results sections of the papers. The rule-based chunker slices documents into fixed-size windows, and in several cases the top-ranked chunks come from Abstract or Discussion sections and not from Results. The numbers are in the corpus but not in the retrieved evidence passed to the synthesizer.

### Finding 4: Q01 and Q02 are stuck — and it is not the synthesizer's fault

The Alzheimer's blood biomarker queries (Q01, Q02) score 5/10 even with Claude. The
judge's rationale for Q01 with the LLM synthesizer:

> *"The answer explicitly acknowledges the absence of specific performance data —
> plasma p-tau217 and Aβ42/40 ratio are mentioned as examples of what is missing from
> the excerpts. Directional agreement and quantitative detail are absent."*

Claude is accurately reporting what the retrieved chunks contain. The chunks are from the introduction and discussion sections, which establish the importance of the topic but do not report the actual diagnostic performance numbers. No amount of synthesis improvement fixes this — the fix is upstream: better chunking or section-aware retrieval that preferentially pulls from Results sections for diagnosis-intent queries.

### Finding 5: Perfect 10/10 scores cluster around clean quantitative documents

Three queries score 10/10 with Claude (Q08, Q09, Q14). All three draw from documents
with clearly structured results sections and explicit effect sizes. The cardiology
meta-analysis (resveratrol) is particularly well-structured. It reports pooled WMD
values with confidence intervals and p-values in dedicated paragraphs that chunk cleanly.

The three dimensions (Source Attribution, Evidence Strength, Uncertainty Calibration) are all achieved at once for these queries: the LLM synthesizer uses [N] citations, names the meta-analysis study design explicitly, and applies appropriately assertive language because the pooled RCT evidence is strong. Structure upstream → full marks downstream.

The lesson: document quality and structure upstream determines answer quality downstream.

---

## What Comes Next

The eval identified three concrete improvement directions, ordered by expected impact:

**1. Section-aware retrieval for diagnosis-intent queries.**
When `QueryAnalyzer` classifies a query as `diagnosis`, the retrieval pipeline should up-weight chunks from Results and Methods sections. The chunk metadata already stores the section label. It just is not used as a retrieval signal yet.

**2. Late chunking or larger chunk size for statistical content.**
The current fixed-size chunker often splits a results paragraph across chunk boundaries, separating the finding from its effect size. Late chunking (embed the full document first and then chunk) or simply increasing chunk size for numeric-heavy sections would keep the statistics with their context.

**3. Query decomposition for multi-part queries.**
Q12 (TKI vs ICI comparison) asks about both TKIs *and* ICIs, and the judge notes that progression-free survival is never explicitly mentioned despite being in the document. A query decomposition step that breaks multi-entity comparison queries into sub-queries and merges the results would surface more of the relevant evidence.

---

## Takeaways

Using an LLM as a judge is not just another benchmark. It measures something that
retrieval metrics structurally cannot i.e. whether the final answer is clinically
actionable. The eight-dimension rubric makes failure modes explicit and attributable:
you can see whether a low score is a retrieval failure (wrong document), a chunking
failure (right document, wrong section), or a synthesis failure (right evidence, vague conclusion). The three dimensions add a safety layer: source attribution catches unsourced assertions, evidence strength catches unsupported conclusions, and uncertainty calibration catches both false confidence and excessive hedging.

The results here confirm an intuition that many RAG practitioners share but rarely
quantify: retrieval is the easier half of the problem. Once the right document is found, turning it into a precise, directional, calibrated clinical statement — with inline citations, an explicit study design, and appropriate confidence language — is where the real difficulty lies. The LLM synthesizer scores 7.50/10 vs the rule-based 5.40/10. That 2.10-point gap represents the quantified value of synthesis over extraction.

The full implementation: rubric, reference claims, judge, and CLI can be found in `evals/` alongside the retrieval eval. Run both together with:

```bash
python evals/answer_eval.py --with-retrieval --verbose
python evals/answer_eval.py --llm --with-retrieval --verbose
```

The gap between the two runs is the quantified value of LLM synthesis over rule-based extraction.

---

*Code: [github.com/ricchasethi/rag-biomedical-decision](https://github.com/ricchasethi/rag-biomedical-decision)*
