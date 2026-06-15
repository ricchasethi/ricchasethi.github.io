---
layout: post
title: "BioRAG: Part 7"
series: "BioRAG"
part: 7
date: 2026-06-15
---

# Building BioRAG: A Decision-Support RAG System for Biomedical Literature

**Part 7 — From Keyword Matching to Meaning: Embeddings and Vector Databases**

---

For the first six parts of this series, BioRAG found documents using a single technique: **keyword matching**. It worked well as seen from evals in Part 5. The right document landed at rank 1 most of the time.

But keyword matching has a blind spot which does matter in medicine. This post is about what the blind spot is, the technology that fixes it i.e. **embeddings** and **vector databases** and how they were added to BioRAG without throwing away the keyword search that already worked.

If you have never thought about how a search engine decides what is "relevant", this post is written for you. No prior knowledge of retrieval is assumed.

---

## The Story So Far: How BM25 Finds Documents

When you type a question into BioRAG, the first thing it does is find candidate documents from the corpus. Until now, it did this with an algorithm called **BM25**.

BM25 is a very sophisticated word-counter. When you ask *"What plasma biomarkers predict Alzheimer's disease?"*, BM25 breaks your question into words/tokens like `plasma`, `biomarkers`, `predict`, `alzheimer`, `disease` and then scans every document for those exact words/tokens. A document that contains "alzheimer" five times and "biomarker" three times scores higher than one that mentions them once.

It is a little smarter than raw counting. BM25 adds two refinements:

- **Rare words count more.** The word "the" appears everywhere, so matching it tells you nothing. The word "neurofilament" appears rarely, so matching it is a strong signal. BM25 down-weights common words and up-weights rare ones.
- **Repetition has diminishing returns.** The tenth mention of "amyloid" adds less than the second. BM25 saturates so one keyword-stuffed document cannot dominate.

For biomedical text, this is genuinely good. Medicine is full of precise terms like gene names (`APOE4`), drug names (`lecanemab`), protein markers (`p-tau217`) etc. When a clinician searches for `APOE4`, they mean *that exact gene* and BM25's exact-match behavior is exactly right. This is why BM25 was the default and why the core engine needs zero external dependencies to run it.

---

## The Blind Spot: Words That Mean the Same Thing

Here is the problem. BM25 matches **words** and not their **meaning**. It has no idea that two different phrases can describe the same thing.

Consider these two phrases:

- *"plasma biomarkers"*
- *"blood-based protein markers"*

To a doctor, these are nearly the same concept. But to BM25, they share **zero words**. If your query says "plasma biomarkers" and the most relevant paper in the corpus happens to phrase it as "blood-based protein markers," BM25 may rank that paper far down the list or even miss it entirely.

This happens constantly in scientific writing because authors deliberately vary their language:

| Your query says | The paper says | Words shared |
|---|---|---|
| heart attack | myocardial infarction | 0 |
| brain scan | neuroimaging | 0 |
| memory loss | cognitive decline | 0 |
| blood sugar | plasma glucose | 0 (well, "plasma"…) |

A keyword search treats every row above as a complete miss. That is the blind spot: **BM25 cannot retrieve a document whose words it does not literally share with the query, no matter how relevant the document is.**

To close this gap, we need to search by *meaning* instead of by *spelling*. That is what embeddings do.

---

## What Is an Embedding?

An **embedding** is a way of turning a piece of text into a list of numbers, a coordinate, such that texts with similar meanings get similar coordinates.

Let me show an example. 

### A toy example you can picture

Imagine we want to place words on a 2-dimensional map. We get to invent two axes:

- **Axis 1 (left ↔ right):** how much the word relates to *the heart*.
- **Axis 2 (down ↔ up):** how much the word relates to *the brain*.

Now we place some medical words on this map by giving each one two numbers — `[heart-score, brain-score]`:

- `alzheimer` → `[0.1, 0.9]` — barely about the heart, very much about the brain.
- `memory` → `[0.2, 0.8]` — also brain-ish. **Notice it sits close to `alzheimer`.**
- `cardiac` → `[0.9, 0.1]` — very heart, not brain. **Far from `alzheimer`.**
- `stroke` → `[0.6, 0.6]` — it involves both, so it sits in the middle.

Now the magic: to find words related to `alzheimer`, we do not look for the *letters* a-l-z-h… We look for the *nearest points on the map*. `memory` is close so it is related, **even though it shares no letters with "alzheimer."** `cardiac` is far, so it is unrelated.

That is an embedding. Each word (or sentence, or paragraph) becomes a point in space and **distance in that space means difference in meaning**. Close together = similar meaning. Far apart = different meaning.

### From 2 dimensions to 768

My toy map had 2 axes that I picked by hand. Real embedding models do two things differently:

1. **They use many more dimensions** — hundreds or thousands. The model BioRAG produces **768 numbers** per text. You cannot picture 768-dimensional space, but the math of "which points are close" works exactly the same.
2. **Nobody hand-picks the axes.** The model *learns* them by reading enormous amounts of text. After training, some dimensions end up loosely corresponding to concepts like "is this about oncology," "is this a statistical result," "is this past tense".But, mostly they are abstract directions the model found useful. 

The payoff: when a model has read millions of biomedical sentences, it learns that "plasma biomarkers" and "blood-based protein markers" appear in the same contexts. So it places them at nearly the same coordinates. Search by nearness and you find one when you ask for the other. **The blind spot is closed.**

### How "nearness" is actually measured

When BioRAG compares two embeddings, it uses **cosine similarity**. It is a number from -1 to 1 that measures whether two coordinate-arrows point in the same direction. 1 means "same meaning," 0 means "unrelated," negative means "opposite." You do not need the formula; just hold onto the intuition: *higher cosine = closer in meaning.*

The key idea is that cosine measures the **angle** between two arrows drawn from the origin to each point. Two arrows pointing the same way are "similar" no matter how long they are. The figure below shows it on both sides: the keyword blind spot on the left (same meaning, zero shared words), and on the right the same four medical words drawn as vectors from the origin — `alzheimer` and `memory` almost overlap (a tiny angle), while `cardiac` points off in a very different direction (a wide angle):

![From Spelling to Meaning: Embeddings & Cosine Similarity](/blog/linkedin_post_7a_diagram.png)

`alzheimer` and `memory` point almost the same direction, so the angle between them is tiny and the cosine is near 1. `alzheimer` and `cardiac` point in nearly opposite corners of the space, so the angle is wide and the cosine is low — even though both arrows are the same length. Working through all three comparisons against `alzheimer`:

| Pair | Angle between arrows | Cosine similarity | Interpretation |
|---|---|---|---|
| alzheimer ↔ memory  | ≈ 8°  | **0.99** | almost identical meaning |
| alzheimer ↔ stroke  | ≈ 39° | **0.78** | related (stroke touches the brain too) |
| alzheimer ↔ cardiac | ≈ 77° | **0.22** | mostly unrelated |

This is exactly the ranking a clinician would expect: `memory` closest to `alzheimer`, `cardiac` furthest and BioRAG gets it purely from the angles between embedding vectors.

---

## Choosing an Embedding Model for BioRAG

Not all embedding models are equal, and for biomedical text the choice matters a lot. A model that learned its coordinates by reading news articles and Reddit will place medical terms clumsily, because it rarely saw them. A model that learned by reading PubMed will place them precisely.

Every model below is free to download from the [Hugging Face](https://huggingface.co/) hub and drops straight into BioRAG's `EmbeddingModel(model_name=...)`. They fall into two families.

**General-purpose retrievers** — trained on broad web text, strong all-rounders, small and fast:

| Model | Dim | Advantages | Disadvantages |
|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | Tiny (~80 MB), very fast, runs comfortably on CPU | Weakest on biomedical jargon; rare gene/drug terms placed loosely |
| `BAAI/bge-small-en-v1.5` | 384 | Lightweight, retrieval-tuned, available via `fastembed` (no `torch`) | General domain; needs a query-instruction prefix to perform its best |
| `BAAI/bge-base-en-v1.5` | 768 | Stronger general retrieval, same dim as the default | ~3× larger than `bge-small`; still not biomedical |
| `intfloat/e5-base-v2` | 768 | Excellent general retrieval benchmarks | Requires `"query:"` / `"passage:"` prefixes or quality drops sharply |
| `BAAI/bge-large-en-v1.5` | 1024 | Top-tier general accuracy | Heavy and slow; GPU strongly recommended; bigger vectors to store |

**Biomedical specialists** — trained on PubMed / clinical text that understand the vocabulary:

| Model | Dim | Advantages | Disadvantages |
|---|---|---|---|
| **`pritamdeka/S-PubMedBert-MS-MARCO`** | **768** | **PubMed + Q&A tuned; strong question→passage matching; the BioRAG default** | **~440 MB; pulls in `torch`; slower than MiniLM** |
| `NeuML/pubmedbert-base-embeddings` | 768 | PubMedBERT packaged as a ready sentence-embedder; great on abstracts | Tuned for similarity, not specifically Q&A retrieval |
| `pritamdeka/S-BioBert-snli-multinli-stsb` | 768 | BioBERT-based, solid biomedical sentence similarity | Older; trained on NLI/STS, not retrieval pairs |
| `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` | 768 | Superb at entity/synonym matching (gene & drug aliases) | Built for entity linking, not passage retrieval - weaker on full questions |

Two practical caveats that trip people up:

- **Raw `BioBERT` / `PubMedBERT` / `ClinicalBERT` checkpoints are *not* embedding models.** They are masked-language models. Without sentence-transformer fine-tuning they produce poor coordinates. Always pick a checkpoint that is explicitly packaged for embeddings (like the four above), not the base model.
- **Switching to a model with a different dimension means re-embedding the whole corpus.** BioRAG sizes its Qdrant collection to the model's output dim (e.g. 768), so moving to a 384- or 1024-dim model requires a fresh collection. This means delete `./qdrant_data` and let it rebuild.

BioRAG defaults to **`S-PubMedBert-MS-MARCO`** for two reasons that compound:

1. **It read the right material.** It was trained on PubMed, so it has seen "p-tau217," "amyloid-β," and "neurofilament light chain" in their natural habitat. It knows they cluster near "Alzheimer's" and not near "lung cancer."
2. **It was tuned for question-answering.** The "MS-MARCO" part means it was specifically trained to match *questions* to *relevant passages*. This is exactly what a search engine does. Many embedding models are trained to match sentences to similar sentences; this one was trained to match a question to its answer.

### Which model for which scenario?

| Your situation | Pick | Why |
|---|---|---|
| Default — best accuracy on PubMed abstracts | `S-PubMedBert-MS-MARCO` | Domain + question→passage tuning; what BioRAG ships with |
| CPU-only box, fast prototyping, demos | `all-MiniLM-L6-v2` | Smallest and fastest; accept some biomedical fuzziness |
| Want to avoid installing `torch` | `BAAI/bge-small-en-v1.5` via `fastembed` | Runs without the heavy deep-learning stack |
| Queries hinge on gene/drug **synonyms & aliases** | `SapBERT-from-PubMedBERT-fulltext` | Purpose-built to map entity variants to the same point |
| GPU available, accuracy matters most | `bge-large-en-v1.5` or a large biomedical model | Highest ceiling; the cost is speed and storage |
| Mixed biomedical + general/clinical corpus | `NeuML/pubmedbert-base-embeddings` | Strong on abstracts without over-specializing |

The model is swappable in one line of code, so a deployment that cannot afford the heavier PubMed model can drop to the lighter `bge-small` and trade some accuracy for speed:

```python
from hybrid_retrieval import EmbeddingModel, DenseRetriever
from core.rag_engine import BioRAGEngine

# Default: the PubMed-tuned model
engine = BioRAGEngine(dense_retriever=DenseRetriever(EmbeddingModel()))

# Lighter alternative — same interface, smaller footprint
dr = DenseRetriever(EmbeddingModel("BAAI/bge-small-en-v1.5"))
engine = BioRAGEngine(dense_retriever=dr)
```

A practical note on cost: the embedding model is a neural network that takes real time to run. Embedding the entire corpus once is the expensive step. We do not want to repeat it on every restart and this brings us to where those coordinates live.

---

## Where Embeddings Live: The Vector Database

Once every chunk of every document has been turned into 768 numbers the problem is now its storage. When a query arrives, we turn *it* into 768 numbers too, and then we need to answer:

> *Of the thousands of stored coordinates, which ones are nearest to this query's coordinate?*

Doing this by brute force i.e. comparing the query against every single stored vector works for a few hundred documents but becomes painfully slow as the corpus grows. We need a database built specifically for "find me the nearest points in high-dimensional space." That is a **vector database**.

BioRAG uses **Qdrant**. A vector database does three jobs an ordinary database cannot:

1. **It stores vectors efficiently** : thousands of 768-number coordinates, each tagged with which chunk and document it came from.
2. **It searches by nearness quite fast.** Instead of checking every vector, it uses clever indexing (think of it as a shortcut map of the space) to find the closest matches in a fraction of the time. This is called *approximate nearest neighbor* search. It is "approximate" because it trades a tiny, usually invisible amount of accuracy for a huge speed gain.
3. **It persists to disk.** The expensive embedding step happens once. The coordinates are saved so a restart reloads them instantly instead of re-embedding the whole corpus.

In BioRAG, the vector database can run three ways, depending on what you are doing:

| Mode | When | Why |
|---|---|---|
| In-memory | Tests and evals | Fresh every run, nothing left on disk |
| File-based (default) | Development | Embeddings saved to `./qdrant_data`, survive restarts |
| Server | Production | A shared Qdrant service many processes can query |

There is one subtlety worth calling out because it is the kind of thing that quietly wastes money. Every time BioRAG starts, it re-loads its sample corpus. Without a guard, it would re-embed every chunk on every launch and one has to pay for expensive neural-network cost over and over for text it has already processed. So before embedding, BioRAG asks Qdrant *"which of these chunks do you already have?"* and embeds only the genuinely new ones. The result: the corpus is embedded **once**, and only freshly ingested papers ever touch the model again.

---

## Bringing It Together: Two Searches Are Better Than One

Here is the design decision that ties the whole post together. I run **both** BM25 and embeddings. 

Why keep the old keyword search if embeddings understand meaning? Because the two techniques fail in *opposite* ways and that is exactly what you want from a pair:

- **BM25 is precise but literal.** It nails exact terms like `APOE4`, `lecanemab`,but misses paraphrases.
- **Embeddings are flexible but fuzzy.** They catch paraphrases like "memory loss" ≈ "cognitive decline". However, it can occasionally drift toward something that is topically close yet wrong and they can fumble a rare gene ID that the keyword search would have matched perfectly.

When one is weak, the other tends to be strong. So BioRAG sends the query to both, gets two ranked lists, and merges them. The merging method is called **Reciprocal Rank Fusion**. Now the all-important question of *whether running both actually produces better answers for Alzheimer's queries* is the entire subject of the next post.

Putting the whole back half of this post together — picking a domain-aware model, storing its vectors in a database that embeds once, and running both searches side by side:

![Choosing an Embedding Model + Where Vectors Live](/blog/linkedin_post_7b_diagram.png)

Crucially, the dense retriever is **optional and bolted on**. With no embedding model configured, BioRAG behaves uses pure BM25, zero external dependencies. Add a `dense_retriever` and the meaning-based search switches on. 

---

## Takeaways

- **Keyword search (BM25) matches spelling, not meaning.** It is excellent for precise medical terms but blind to paraphrases and scientific writing is full of paraphrases.
- **An embedding turns text into a coordinate** in a high-dimensional space where *distance means difference in meaning*. Search becomes "find the nearest points" instead of "find the matching letters." That is how "memory loss" can retrieve a paper about "cognitive decline."
- **The embedding model must know the domain.** BioRAG uses `S-PubMedBert-MS-MARCO`, trained on PubMed and tuned for question-answering, because a model that has actually read biomedical literature places its terms far more accurately than a general-purpose one.
- **A vector database (Qdrant) stores those coordinates and searches them fast,** persisting to disk so the expensive embedding step happens only once.
- **The winning move is not replacing BM25 but combining it with embeddings,** because the two methods fail in opposite ways.

In next blog I will discuss whether that combination actually *helps* on real Alzheimer's queries or not.

---

*Code: [github.com/ricchasethi/rag-biomedical-decision](https://github.com/ricchasethi/rag-biomedical-decision)*
