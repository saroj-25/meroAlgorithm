# MeroAlgorithm


**An AI-powered DSA tutor that teaches algorithms and data structures to Nepali students
in their own language — Romanized Nepali mixed with English.**


An end-to-end implementation of *"Natural Language Processing-Driven Chatbot for Algorithm
Learning: A Retrieval-Augmented Generation Approach using Romanized Nepali and English"*
(Bhandari & Dhital, *Aadim Journal of Multidisciplinary Research Information & Technology*
2(1), 132–149, 2026) — built to be read, run and extended, not just described.



## Runs anywhere: the degradation ladder


| Component | Paper-faithful backend | Offline fallback |
|---|---|---|
| Encoder | `paraphrase-multilingual-mpnet-base-v2` (768-d) | char n-gram TF-IDF + truncated SVD |
| Vector index | FAISS HNSW (M=32, efC=200, efS=64) | exact numpy cosine |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | dense score + IDF overlap + char-trigram Dice |
| Generator | Llama-3.1-8B-Instruct (Ollama) / GPT-4o-mini | extractive template generator |
| Semantic metric | BERTScore | token-F1 proxy (labelled as such, never as BERTScore) |
| Tracking | MLflow | JSONL run log |

---

## Pipeline

```
query ─► normalize ─► code-switch detection (char n-gram classifier)
        │
        ├── english          → as-is
        ├── romanized nepali → "roman [SEP] devanagari" expansion
        └── code-mixed       → as-is (splitting degraded retrieval in the paper's pilots)
        │
        ├─ dense top-50 ─┐
        ├─ BM25 top-50 ──┴─► reciprocal rank fusion (k=60) ─► top-20
        │                                                       │
        │                              cross-encoder re-rank ◄──┘ → top-5
        │
        └─► pedagogical prompt (grounding · register · intuition→formal→code→complexity
            · one Socratic check-back) ─► generator ─► citation validation ─► answer
```

Example (offline generator, no LLM):

```
you > yo dijkstra ko algorithm negative weight ma kina fail huncha

[register: code_mixed | nepali=44%, english=11%, ambiguous=44% | 0.65s | template]

Tapaile **Dijkstra** (Graphs and Traversals) ko barema sodhnu bhayo. ...
**Formal statement (exam ma lekhne):** Correct only for non-negative weights, because
settling assumes no later path can be cheaper. [GRF-EN-0023]
**Complexity:** O((V + E) log V) with a binary heap. [GRF-EN-0023]
**Source chunks:** [GRF-CM-0007], [GRF-EN-0023], ...
**Ek prashna tapailai:** Yo complexity kun case ko ho — best, average ki worst? Kina?
```

Citations are validated against the retrieved set, so a fabricated source id is detectable
(and is asserted against in the test suite).

---

## Verified results

Reproduced statistics (synthetic study data calibrated to the paper):

| Quantity | Paper | This repo |
|---|---|---|
| Gain difference | 11.7 pp (95% CI 6.2–17.2) | 11.7 pp (95% CI 6.1–17.3) |
| t(50), Cohen's d | 4.21, 1.17 | 4.22, 1.17 |
| ANCOVA adjusted difference | 12.1 (CI 7.4–16.8) | 12.0 (CI 7.0–17.0) |
| Shapiro–Wilk / Levene | all satisfied | all satisfied |
| Code-mixed share of logs | 43.1% | 43.1% |
| Knowledge base | 1,252 chunks | 1,252 chunks |

Retrieval ablation on the synthetic corpus (P@5) — note the **disagreement** with the
paper, which is analysed in `docs/findings.md`:

| Retriever | English | Romanized Nepali | Code-mixed |
|---|---|---|---|
| dense only | 0.91 | 0.85 | 0.84 |
| BM25 only | 0.84 | 0.51 | 0.49 |
| hybrid + RRF | 0.89 | 0.64 | 0.60 |
| hybrid + RRF + re-rank | 0.91 | 0.74 | 0.68 |
| hybrid, weighted RRF (w=0.4) + re-rank | 0.91 | 0.78 | 0.76 |

Unweighted RRF gives an equal vote to a retriever that is far weaker on the query class,
so it imports BM25's errors on Romanized Nepali. The takeaway is a practical rule:
**measure retriever strength per query class before fusing, and use weighted RRF when they
differ.**

Language-ID classifier: held-out macro-F1 0.986 on the generated corpus (the paper reports
0.91 on real chat logs; ours is easier because labels come from a transparent rule set).

---

## Layout

```
configs/          default.yaml · ablation.yaml · hparams.yaml   (a run = code + config + seed)
src/data/         concept registry, KB builder, eval sets, study simulator, ingest, loaders
src/preprocessing/ lexicons, normalization, transliteration, language ID, chunking, query branches
src/models/       encoders, vector stores, BM25, re-rankers, prompts, generators, losses
src/retrieval/    index builder, hybrid retriever + reciprocal rank fusion
src/training/     numpy mini-batch trainer (forward/backward/val/early stop), train_lid CLI
src/evaluation/   retrieval + generation metrics, stats tests, three evaluation scripts
src/inference/    RAG pipeline, interactive CLI
experiments/      run_all.py (full reproduction + REPORT.md), hyperparameter_search.py
streamlit_app/    Dataset · Training · Chat · Results
tests/            32 tests: BM25, metrics, preprocessing, statistics, end-to-end
docs/             paper_analysis.md · architecture.md · reproduction.md · findings.md
```

## Citation

```bibtex
@article{bhandari2026algosathi,
  title  = {Natural Language Processing-Driven Chatbot for Algorithm Learning:
            A Retrieval-Augmented Generation Approach using Romanized Nepali and English},
  author = {Bhandari, Saroj and Dhital, Puja},
  journal= {Aadim Journal of Multidisciplinary Research Information \& Technology},
  volume = {2}, number = {1}, pages = {132--149}, year = {2026}
}
```

This implementation is an independent reproduction for learning and research. Licence: MIT.