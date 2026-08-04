# Architecture and repository guide

## 1. Pipeline

```
                      OFFLINE (once per corpus)
 knowledge_base.jsonl ──► chunking ──► encoder ──► vector index (FAISS HNSW / numpy)
                                   └─► BM25 tokenizer ──► inverted index

                      ONLINE (per query)
 student query
   │
   ├─ normalize (NFKC, repeat collapsing)
   ├─ code-switch detection (char n-gram classifier → token labels)
   ├─ branch:  english → as-is
   │           romanized nepali → "roman [SEP] devanagari" expansion
   │           code-mixed → as-is (never split)
   │
   ├─ dense top-50 ─┐
   ├─ BM25 top-50 ──┼─► reciprocal rank fusion (k=60) ─► top-20
   │                │
   ├────────────────┴─► cross-encoder re-rank ─► top-5
   │
   ├─ pedagogical prompt (grounding + register + 4-stage scaffold + Socratic)
   ├─ generator (Ollama / OpenAI / offline extractive)
   └─ post-process: citation validation, latency, log record
```

## 2. Degradation ladder

Every heavy component has a lightweight substitute, selected automatically. This is why
`make all` works on a laptop with no GPU and no network.

| Component | Paper-faithful | Offline fallback | Config key |
|---|---|---|---|
| Encoder | `paraphrase-multilingual-mpnet-base-v2` | char n-gram TF-IDF + truncated SVD | `encoder.backend` |
| Vector index | FAISS HNSW (M=32) | exact numpy cosine | `vector_store.backend` |
| Re-ranker | `ms-marco-MiniLM-L-6-v2` cross-encoder | dense score + IDF overlap + char-trigram Dice | `retrieval.reranker.backend` |
| Generator | Llama-3.1-8B-Instruct / GPT-4o-mini | extractive template generator | `generation.backend` |
| Tracking | MLflow | JSONL run log | `tracking.backend` |
| Semantic metric | BERTScore | token-F1 proxy (labelled as such) | automatic |

Set `project.mode: offline` to force the fallbacks even when the heavy libraries exist.

## 3. Repository layout

```
algosathi/
├── README.md                  quickstart, results, honesty notice
├── Makefile                   `make data`, `make index`, `make train`, `make all`, `make app`
├── requirements.txt           core dependencies (project runs fully on these)
├── requirements-full.txt      optional paper-faithful stack (torch, faiss, ...)
│
├── configs/
│   ├── default.yaml           the single source of truth for a run
│   ├── ablation.yaml          retriever grid for Table 3
│   └── hparams.yaml           sweep space for experiments/hyperparameter_search.py
│
├── data/
│   ├── raw/                   drop your own PDFs / notes here
│   ├── processed/             generated: knowledge_base.jsonl, eval_queries.jsonl,
│   │                          lid_dataset.jsonl, study_scores.csv, interaction_logs.csv
│   └── external/              third-party corpora, if you add any
│
├── notebooks/
│   └── 01_quickstart.ipynb    load the index, run a query, inspect retrieval internals
│
├── src/
│   ├── data/
│   │   ├── dsa_content.py             143-concept DSA registry in a compact DSL
│   │   ├── build_knowledge_base.py    expands it to the exact Table 1 composition
│   │   ├── build_eval_sets.py         300 annotated queries + 4k LID sentences
│   │   ├── simulate_study_data.py     study scores / satisfaction / logs, calibrated
│   │   └── loaders.py                 typed loaders used by scripts and the app
│   │
│   ├── preprocessing/
│   │   ├── lexicons.py         Romanized Nepali / English word lists, technical regexes
│   │   ├── normalize.py        NFKC, zero-width stripping, repeat collapsing
│   │   ├── transliterate.py    greedy Roman → Devanagari (indic-transliteration if present)
│   │   ├── language_id.py      CharNGramLID model + rule-based fallback
│   │   ├── chunking.py         250–450 token chunking with 50-token overlap
│   │   └── query_pipeline.py   the three branches of Section 3.3
│   │
│   ├── models/
│   │   ├── encoders.py         encoder interface + both backends
│   │   ├── vector_store.py     FAISS HNSW + numpy flat index
│   │   ├── bm25.py             self-contained BM25Okapi
│   │   ├── reranker.py         cross-encoder + offline blend re-ranker
│   │   ├── prompts.py          the pedagogical system prompt ("Appendix A")
│   │   ├── generator.py        Ollama / OpenAI / extractive generators
│   │   └── losses.py           softmax cross-entropy (+grad), L2, InfoNCE
│   │
│   ├── retrieval/
│   │   ├── index_builder.py    offline phase: build and persist all indexes
│   │   └── hybrid.py           RRF fusion + HybridRetriever (ablation switches)
│   │
│   ├── training/
│   │   ├── trainer.py          numpy mini-batch trainer: forward, backward, val, early stop
│   │   └── train_lid.py        CLI: trains the code-switch classifier, writes curves
│   │
│   ├── evaluation/
│   │   ├── retrieval_metrics.py   P@k, R@k, MRR, nDCG
│   │   ├── generation_metrics.py  ROUGE-L, token-F1, grounding, citation, register match
│   │   ├── stats_tests.py         t-test, Cohen's d, ANCOVA, Shapiro, Levene, power
│   │   ├── evaluate_retrieval.py  Table 3 + per-query error analysis
│   │   ├── evaluate_generation.py Table 4
│   │   └── evaluate_learning.py   Tables 5–7 (also accepts a real scores CSV)
│   │
│   ├── inference/
│   │   ├── rag_pipeline.py     AlgoSathiRAG — the deployed system
│   │   └── cli.py              interactive terminal chat
│   │
│   ├── visualization/plots.py  every matplotlib figure written to results/figures
│   └── utils/                  config, logging, seeding, io, experiment tracking
│
├── experiments/
│   ├── run_all.py              full reproduction + results/REPORT.md
│   ├── hyperparameter_search.py sweeps rrf_k, depths, fusion weights, BM25 k1/b
│   └── runs/                   JSONL run log (or mlruns/ when MLflow is installed)
│
├── checkpoints/                lid_model.pkl, index/{encoder,vector_store,bm25}.pkl
├── results/
│   ├── tables/                 table3..table7 CSVs, per-query CSVs, JSON analyses
│   ├── figures/                PNGs
│   ├── logs/                   run logs
│   └── REPORT.md               generated side-by-side comparison with the paper
│
├── streamlit_app/
│   ├── app.py                  home + pipeline status
│   └── pages/                  1_Dataset, 2_Training, 3_Chat, 4_Results
│
├── tests/                      pytest suite (BM25, metrics, preprocessing, stats, e2e)
└── docs/
    ├── paper_analysis.md       Phase 1: the full reading of the paper
    ├── architecture.md         this file
    ├── reproduction.md         exact commands and expected outputs
    └── findings.md             where the reproduction agrees and disagrees
```

## 4. Data contracts

**Knowledge-base chunk**

```json
{"chunk_id": "TRE-EN-0042", "topic": "Trees and BSTs", "concept": "lca",
 "concept_name": "Lca", "facet": "complexity", "language": "en",
 "difficulty": "medium", "source": "lecture_slides", "text": "...", "n_tokens": 312}
```

`chunk_id` is `TOPIC-REGISTER-INDEX`; register is `EN` or `CM`. Citations in generated
answers are validated against these ids, which is how "no fabricated sources" is enforced.

**Evaluation query**

```json
{"query_id": "CO-017", "query": "yo dijkstra ko time complexity kasari nikalne",
 "query_type": "code_mixed", "concept": "dijkstra", "topic": "Graphs and Traversals",
 "relevant_chunk_ids": ["GRF-EN-0007", "..."], "n_relevant": 11}
```

**Study scores** (`student_id, group, pre_test, post_test`) — the only contract you need
to satisfy to analyse real classroom data with `evaluate_learning.py`.

## 5. Extending the system

* **Your own course material** → `python -m src.data.ingest --input data/raw` or the
  Dataset page's uploader, then rebuild the index.
* **A different LLM** → implement `BaseGenerator.generate` and register it in
  `build_generator`. The prompt does not change; that is the point of the abstraction.
* **A different encoder** → implement `BaseEncoder.encode`; everything downstream is
  dimension-agnostic.
* **A new retrieval strategy** → add a config dict to `configs/ablation.yaml`; the
  evaluation grid picks it up with no code change.
