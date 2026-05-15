# Building BioRAG: A Decision-Support RAG System for Biomedical Literature

**Part 1 of 5 — Introduction, Query Understanding, and Retrieval**

---

## Why Biomedical Literature Needs a Different Approach

The biomedical literature problem is unlike any other information retrieval problem. PubMed indexes over 35 million citations. A working clinician or researcher asking "which plasma biomarkers best predict early Alzheimer's disease?" is not looking for the ten most popular documents about Alzheimer's or biomarkers — they are looking for specific, evidenced, traceable answers that can inform a real decision.

General-purpose RAG systems fall short here for three reasons.

**Precision matters more than recall.** A chatbot that confidently synthesises an answer from tangentially relevant papers is worse than useless in a clinical context — it is dangerous. An answer needs to be traceable to the specific passage that supports it, with an explicit statement of how strong that support is.

**The vocabulary is highly specialised.** Biomedical text is dense with abbreviations (PD-L1, APOE4, pTau217), statistical notation (AUC, HR, CI), and entity names that carry the entire discriminative signal for a query. Semantic embeddings trained on general corpora can miss the distinction between "AD biomarkers" and "cardiovascular biomarkers" because both look superficially similar in vector space.

**Uncertainty must be surfaced, not buried.** When the corpus does not contain enough evidence to answer a question confidently, the system must say so. Calibrated uncertainty is a feature, not a limitation.

BioRAG was built to address all three. It is a retrieval-augmented generation engine that returns not just an answer, but a structured decision output: an evidence classification, an explicit reasoning chain, a confidence score, and a list of knowledge gaps.

---

## The Series Roadmap

This is the first of five posts. Here is where we are going:

| Post | Topic |
|------|-------|
| **1 (this post)** | Introduction · Query Analyzer · BM25 Retrieval |
| 2 | Reranking · Evidence Classification · False Positive Suppression |
| 3 | Knowledge Gap Detection · Confidence Scoring · Reasoning Chains |
| 4 | LLM Answer Synthesis · Grounding · Prompt Design |
| 5 | Evaluation · MRR · NDCG · Lessons from Building Retrieval Evals |

---

## The Pipeline at a Glance

Before going deep, here is the full pipeline so you can orient each post within it:

```
Query
  │
  ▼
QueryAnalyzer          ← intent, entities, expanded tokens
  │
  ▼
InvertedIndex (BM25)   ← top-K candidate chunks            ← this post
  │
  ▼
Reranker               ← section-aware + entity penalty    ← Post 2
  │
  ▼
EvidenceClassifier     ← direct / indirect / contradictory ← Post 2
  │
  ▼
KnowledgeGapDetector   ← missing data, contradictions      ← Post 3
  │
  ▼
AnswerSynthesizer      ← reasoning chain + confidence      ← Posts 3 & 4
  │
  ▼
DecisionOutput         ← structured, auditable answer
```

Each stage is a separate class with a single responsibility. The design is intentional: you can upgrade any one stage — swap BM25 for dense embeddings, or replace the rule-based synthesizer with an LLM — without touching the rest of the pipeline.

---

## Stage 1: Query Analyzer

The `QueryAnalyzer` is the first stage in the pipeline, and its job is deceptively simple: turn a natural language question into a structured representation the rest of the pipeline can act on. In practice, getting this right is most of the work.

### Intent Classification

Not all biomedical questions are the same. "What is the mechanism of PD-L1 inhibition?" asks for a biological explanation. "Which treatment is better: ceftazidime-avibactam or colistin for CRE?" asks for a comparison. "What is the 5-year survival rate for stage III NSCLC?" asks for a prognostic figure.

These different intents should retrieve from different parts of a paper and weight different evidence types differently. A mechanism query should prioritise Methods and Results sections. A treatment comparison should prioritise Results and Discussion. A definition query should prioritise the Introduction and Abstract.

BioRAG classifies queries into seven intent types — `mechanism`, `comparison`, `treatment`, `diagnosis`, `prognosis`, `epidemiology`, and `general` — by matching trigger phrases against the query. This drives section weighting in the reranker downstream.

**What this looks like in practice:**

```
Query:   "What biomarkers predict Alzheimer's disease?"
Intent:  diagnosis
Entities: ["Alzheimer"]

Query:   "Compare ceftazidime-avibactam versus colistin for CRE"
Intent:  comparison
Entities: ["CRE", "Ceftazidime-avibactam", "Colistin"]
```

**Key parameter: trigger phrase coverage.** If a query does not match any trigger phrase it falls back to `general`, losing all the intent-specific section boosting downstream. Regularly audit misclassified queries from your eval set against the trigger phrase list.

**How to improve it:** The trigger-phrase approach is brittle. Replacing it with a small fine-tuned classifier (even a logistic regression over TF-IDF features trained on a few hundred labelled queries) lifts accuracy significantly with minimal infrastructure cost.

### Entity Extraction

Biomedical queries are almost always anchored by one or two named entities — a disease, a gene, a drug, a pathogen. These entities are the discriminative signal that separates "Alzheimer's biomarkers" from "cardiovascular biomarkers" even when both queries contain the word "biomarker" many times over.

The `QueryAnalyzer` extracts entities by identifying capitalised multi-word terms, filtered against a blocklist of question words (`What`, `Which`, `How`…). This is intentionally lightweight — no NER model, no external API.

**Key limitation: case sensitivity.** The entity extractor relies on capitalisation. A query typed entirely in lowercase — `alzheimer's disease biomarkers` — yields no entities. This matters because entity presence drives the false-positive suppression penalty in the reranker (covered in Post 2). We address this with a `_GENERIC_BIO_TERMS` approach rather than entity-only logic, but a case-insensitive NER model would be the cleaner long-term fix.

**How to improve it:** Integrate a lightweight biomedical NER model such as [PubMedBERT-NER](https://huggingface.co/pruas/BENT-PubMedBERT-NER-Gene) or the scispaCy `en_ner_bc5cdr_md` pipeline (covers diseases and chemicals). Even a dictionary lookup against the UMLS Metathesaurus would lift entity recall dramatically for common biomedical terms.

### Abbreviation Expansion

Biomedical abbreviations are a retrieval hazard. A query containing "PD-L1" should also match documents that spell it out as "programmed death-ligand 1". A query about "PCR" should match "polymerase chain reaction". Without expansion, these queries silently miss relevant passages.

BioRAG maintains an `ABBREV_MAP` that expands common abbreviations at query time, appending the expanded tokens to the query token list. The original abbreviation is preserved so both short and long forms match.

```python
# In QueryAnalyzer.analyze()
expanded_tokens = list(tokens)
for token in tokens:
    if token.lower() in self.processor.ABBREV_MAP:
        expansion = self.processor.tokenize(self.processor.ABBREV_MAP[token.lower()])
        expanded_tokens.extend(expansion)
```

**How to improve it:** The current map is hand-curated and covers around 15 common abbreviations. The right approach at scale is to extract abbreviation-definition pairs directly from the corpus using the Schwartz-Hearst algorithm — the pattern `term (ABBREV)` appears in nearly every methods section. This means the expansion map grows automatically with whatever literature you ingest, at zero maintenance cost.

---

## Stage 2: BM25 Retrieval

Once the query is analysed, the `InvertedIndex` retrieves the top-K candidate chunks using BM25 — specifically the Okapi BM25 variant.

### Why BM25 and Not Dense Embeddings?

This is the most common question about this system, so let me address it directly.

Dense embedding models (sentence-transformers, OpenAI text-embedding-3) encode semantic similarity. They are excellent at finding documents that *mean* the same thing even when they use different words. For general-domain retrieval this is a significant advantage.

For biomedical retrieval, it becomes a liability in two ways.

First, the vocabulary is so specialised that semantic proximity in embedding space can mislead. "APOE4 genotype" and "cardiovascular risk gene" might map to nearby vectors even though they describe entirely different biology. BM25 matches on exact tokens — and in biomedicine, exact tokens are the evidence.

Second, dense embeddings require either a GPU, an external API call, or a large local model. BM25 is pure arithmetic over a term-frequency index. For a system that emphasises zero external dependencies and auditability, this matters. BioRAG's core engine runs with no installs beyond the Python standard library.

That said, BM25 and dense retrieval are **complementary, not competing**. The right production architecture is a hybrid: run both in parallel and merge scores with reciprocal rank fusion. We will return to this in Post 5.

### How BM25 Works

BM25 scores a chunk against a query by summing the contribution of each matching query term, weighted by:

- **IDF (inverse document frequency):** rare terms score higher. "Alzheimer" scores far higher than "biomarker" because it appears in far fewer documents.
- **Term frequency saturation:** each additional occurrence of a term adds diminishing returns, controlled by the parameter `K1`. At `K1 = 1.5`, a term appearing 10 times in a chunk does not score 10× a term appearing once.
- **Document length normalisation:** longer chunks are penalised to prevent them from dominating simply by having more text. The parameter `B` controls this. At `B = 0.75`, a chunk twice the average length is moderately penalised.

The scoring formula for a single term `t` in chunk `d` is:

```
score(t, d) = IDF(t) × (TF(t,d) × (K1 + 1)) / (TF(t,d) + K1 × (1 - B + B × |d| / avgdl))
```

where `|d|` is the chunk length and `avgdl` is the average chunk length across the corpus.

### Key Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `K1` | 1.5 | Controls term frequency saturation. Higher values reward repeated terms more. For biomedical text with high repetition of key terms, 1.2–1.5 is typical. |
| `B` | 0.75 | Controls length normalisation. Higher = shorter chunks are favoured. For chunked text, 0.6–0.8 is reasonable. |
| `retrieval_top_k` | 15 | Number of candidate chunks passed to the reranker. Higher improves recall but increases reranking cost. |
| `chunk_size` | 512 tokens | Larger chunks retain more context but dilute term density. Smaller chunks are more precise but may split key sentences. |
| `chunk_overlap` | 64 tokens | Overlap between adjacent chunks. Prevents key sentences from being split across boundaries. |

**Tuning `K1` and `B`:** The defaults work well but are not optimised for any specific corpus. Run your retrieval eval harness (covered in Post 5) with a grid search over `K1 ∈ {1.0, 1.2, 1.5, 2.0}` and `B ∈ {0.5, 0.65, 0.75, 0.9}`. For a corpus dominated by short PubMed abstracts, a lower `B` (0.5) tends to perform better because length variation between chunks is small. For full-text PMC articles with long methods sections, a higher `B` (0.75–0.9) normalises the length advantage those sections get.

**Tuning `chunk_size`:** Smaller chunks (256 tokens) improve precision — the retrieved passage is tightly about one thing — but hurt recall when the key sentence is surrounded by context. Larger chunks (1024 tokens) improve recall but return more noise per hit. 512 with 64-token overlap is a reasonable default for mixed abstract/full-text corpora.

### The False Positive Problem

BM25's fundamental weakness for multi-disease corpora is that it rewards term presence but has no concept of term absence. A query for "Alzheimer's disease biomarkers" tokenises to `[alzheimer, disease, biomarker]`. A lung cancer paper that uses "biomarker" and "disease" repeatedly will score near the top of results — even though "alzheimer" never appears in it.

In a general-domain system, this is tolerable. In a decision-support system where a clinician is relying on the returned evidence nodes, surfacing an irrelevant paper at 75% relevance is a serious quality problem. We observed exactly this after ingesting Alzheimer's papers alongside oncology papers.

The root cause is at the retrieval stage. The fix, however, lives in the reranker — which has access to both the query's entity tokens and the chunk content, and can apply a structured penalty. We will cover this in full in Post 2.

### How to Improve Retrieval

**Short term:**
- Tune `K1` and `B` on your eval set. A 1–2% NDCG gain is achievable with no code changes.
- Expand the `ABBREV_MAP` with domain-specific abbreviations from your target corpus. Every abbreviation you miss is a silent recall failure.
- Implement prefix matching carefully. BioRAG includes a prefix fallback for query terms with no exact match, but short-prefix matching (4 chars) produces false positives — `hemo` matches `heart`. Restrict prefix fallback to terms longer than 6 characters.

**Medium term:**
- **Query expansion via pseudo-relevance feedback (PRF):** fetch the top-3 BM25 results, extract their most discriminative terms by TF-IDF, and add those to the query for a second retrieval pass. This is surprisingly effective for biomedical queries where the user may not know the exact terminology used in the target literature.
- **Synonym expansion via UMLS:** map query entities to their UMLS concept IDs and expand with all synonyms. "Heart attack" and "myocardial infarction" become the same query token family.

**Long term:**
- **Hybrid BM25 + dense retrieval with RRF:** run both retrievers in parallel, normalise their rank lists, and merge with reciprocal rank fusion `score = Σ 1/(k + rank_i)` where `k = 60` is standard. The two signals are complementary: BM25 handles rare exact terms; the encoder handles paraphrases.
- **Learned sparse representations (SPLADE):** a middle ground between BM25 and dense retrieval. SPLADE learns to expand document and query representations into sparse vectors over the vocabulary, preserving the interpretability of BM25 while learning term weights from data.

---

## What the First Two Stages Produce

After `QueryAnalyzer` and `InvertedIndex`, the pipeline has:

```python
query_analysis = {
    "original": "What plasma biomarkers predict Alzheimer's disease?",
    "intent": "diagnosis",
    "entities": ["Alzheimer"],
    "expanded_tokens": ["alzheimer", "plasma", "biomarker", "biomarkers", "predict"],
}

retrieved_chunks = [
    RetrievedChunk(doc_title="Plasma Aβ42/p-Tau217 ratio...",       score=9.72, rank=1),
    RetrievedChunk(doc_title="Diagnostic Blood-Based Biomarkers...", score=8.10, rank=2),
    RetrievedChunk(doc_title="EGFR-Mutant Non-Small Cell Lung...",   score=7.30, rank=9),
    # ... up to retrieval_top_k chunks
]
```

These 15 chunks are raw BM25 candidates. Some are excellent. Some — like that lung cancer paper at rank 9 — are off-topic papers that matched on generic terms. The next stage, the Reranker, is where precision is recovered.

---

## Coming Up in Post 2

In the next post we go deep into the Reranker. We will look at how section weighting (prioritising Results over Introduction for factual queries) interacts with term density scoring — and how the discriminative-token recall penalty was designed to suppress the false positive described above.

We will trace the two-stage debugging process that revealed why common clinical verbs like "predict" and "detect" must be treated as generic rather than discriminative, even when they appear in the query. A lung cancer paper showing up at 65% relevance for an Alzheimer's query after the first fix is a useful lesson in how BM25's term presence bias propagates into the reranker unless you're precise about what "discriminative" means.

We will also cover the `EvidenceClassifier` — how a chunk is labelled as `direct`, `indirect`, or `contradictory`, and why this three-way distinction carries more signal for clinical decision support than a continuous relevance score alone.

---

*BioRAG is built in pure Python stdlib. The core engine — BM25 index, reranker, evidence classifier, synthesizer — has zero external dependencies. PubMed ingestion requires only `requests`. LLM answer synthesis requires only `anthropic`.*
