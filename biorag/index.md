---
layout: page
title: "BioRAG"
permalink: /biorag/
---

A five-part series on building a decision-support RAG system for biomedical literature in pure Python — no embeddings, no external ML dependencies.

The system returns structured, auditable answers with evidence classification (direct / indirect / contradictory), explicit reasoning chains, confidence scoring, and knowledge gap detection.

---

{% assign biorag_posts = site.posts | where: "series", "BioRAG" | sort: "part" %}
{% for post in biorag_posts %}
- [{{ post.title }}]({{ post.url | relative_url }}) — {{ post.date | date: "%B %d, %Y" }}
{% endfor %}
