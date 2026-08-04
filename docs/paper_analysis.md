# Paper analysis — AlgoSathi (Bhandari & Dhital, 2026)

*Aadim Journal of Multidisciplinary Research Information & Technology, 2(1), 132–149.*

---

## 1. Research problem

Nepali undergraduates learning data structures and algorithms (DSA) carry a double
cognitive load: the intrinsic difficulty of the material (abstraction, pointers,
asymptotics) plus a **language mismatch** — instruction and textbooks are in formal
English while students reason, argue and ask doubts in a code-mixed register of
Romanized Nepali and English ("yo recursion ko base case kasari define garne hola?").
General LLM tutors do not fix this: they hallucinate off-syllabus content, and their
retrieval components fail to map a code-mixed question onto the semantically
equivalent English course passage.

## 2. Motivation

Two failure modes have distinct fixes, and the paper argues they must be fixed together:

| Failure | Fix the paper adopts |
|---|---|
| Hallucination / off-syllabus answers | Retrieval-Augmented Generation over instructor-authored material |
| Code-mixed queries not matching English passages | Multilingual sentence encoder + lexical fallback + register-preserving prompt |

The deeper motivation is pedagogical, and is the paper's most interesting claim: the
register a learner may use determines **which questions get asked at all**. Exit
interviews report students asking conceptual "why/when" questions in Nepali that they
would not have attempted to formulate in English.

## 3. Research gap

1. No RAG educational chatbot targets DSA in a code-mixed Romanized-Nepali/English setting.
2. Nepali NLP work concentrates on formal Devanagari and on sentiment/hate-speech
   classification, not educational question answering.
3. Classroom evaluations of LLM tutors run in English-medium contexts and never test
   whether code-mixed input is handled at the level of *pedagogical adequacy*.

## 4. Main contributions

1. A RAG architecture designed for code-mixed queries: language-aware preprocessing,
   multilingual semantic retrieval, register-preserving generation prompt.
2. A curated 1,252-chunk DSA knowledge base with topic / difficulty / language metadata,
   including 184 instructor-authored **code-mixed mini-explanations** of common pitfalls.
3. A 12-week quasi-experimental classroom evaluation (n = 52) with retrieval metrics,
   generation-quality measures and a 17-item CHISM-derived satisfaction instrument.
4. A descriptive analysis of 1,983 logged code-mixed learner interactions.

## 5. Proposed methodology

Retrieval-Augmented Generation with a code-mix-aware front end:

1. **Preprocess** — Unicode normalization; token-level code-switch detection; for pure
   Romanized Nepali, optional transliteration to Devanagari for query expansion.
   Code-mixed queries are deliberately *not* split (splitting degraded retrieval in pilots).
2. **Retrieve** — dense retrieval with `paraphrase-multilingual-mpnet-base-v2` over a
   FAISS HNSW index (top-50) in parallel with BM25 over the same corpus.
3. **Fuse and re-rank** — reciprocal rank fusion (k = 60), then a cross-encoder
   (`ms-marco-MiniLM-L-6-v2`) re-ranks the top-20 down to the top-5.
4. **Generate** — Llama-3.1-8B-Instruct (4-bit AWQ, local) or GPT-4o-mini, constrained by
   a prompt enforcing grounding, citation, register preservation, a four-stage
   pedagogical scaffold (intuition → formalization → code → complexity) and one Socratic
   check-back question.

## 6. System architecture

Two phases (Figure 2 of the paper):

* **Offline** — chunk the corpus (250–450 tokens, 50-token overlap) → encode → FAISS
  HNSW index (M = 32, efConstruction = 200, efSearch = 64); in parallel, BM25 tokenize →
  inverted index.
* **Online** — query → preprocessing → hybrid retriever (50 + 50 candidates) → RRF →
  cross-encoder re-rank (20 → 5) → pedagogical prompt constructor → LLM generator →
  citation post-processing → UI with inline source previews.

Deployment is three-tier: Python 3.11 / LangChain / FastAPI backend, SQLite conversation
store with 30-day retention, React SPA frontend with an explicit "my question contains
Nepali" override. Median end-to-end latency: 1.78 s.

## 7. Algorithm explanation

* **Code-switch detection** — character n-gram classifier emitting per-token probabilities
  over {Romanized Nepali, English, script-ambiguous}; token-level F1 = 0.91. Character
  features are essential because Romanized Nepali has no standard orthography.
* **Dense retrieval** — cosine nearest neighbours in a shared 768-d multilingual space,
  approximated by HNSW (a navigable small-world graph; logarithmic search).
* **BM25** — probabilistic lexical scoring; recovers rare technical strings (AVL,
  Kruskal, Floyd–Warshall) that the encoder under-represents.
* **Reciprocal rank fusion** — rank-based (not score-based) merge, so the two retrievers
  need no score calibration.
* **Cross-encoder re-ranking** — joint query–chunk encoding; more accurate than
  bi-encoder cosine, affordable because only 20 pairs are scored.

## 8. Mathematical formulation

RAG (Lewis et al.):

$$p(y \mid x) = \sum_{z \in \mathcal{Z}} p_\eta(z \mid x)\, p_\theta(y \mid x, z)$$

BM25 (k₁ = 1.5, b = 0.75):

$$\text{score}(q,d) = \sum_{t \in q} \text{IDF}(t)\,\frac{f(t,d)(k_1+1)}{f(t,d) + k_1\left(1-b+b\frac{|d|}{\text{avgdl}}\right)}$$

Reciprocal rank fusion (k = 60):

$$\text{RRF}(d) = \sum_{r} \frac{1}{k + \text{rank}_r(d)}$$

Evaluation: P@5, MRR, R@10 for retrieval; BERTScore F1 for generation; for learning
outcomes an independent-samples t-test on gains with Cohen's d, and ANCOVA
`post ~ group + pre` giving the covariate-adjusted between-group difference.

## 9. Dataset details

| Component | Size | Notes |
|---|---|---|
| Knowledge base | 1,252 chunks | 1,068 English + 184 instructor code-mixed |
| Retrieval eval set | 300 annotated queries | 100 each English / Romanized Nepali / code-mixed; Cohen's κ = 0.81 |
| Language-ID corpus | 4,000 sentences | Romanized Nepali, English, code-mixed |
| Classroom study | 52 students | 26 experimental, 26 control; mean age 19.4 |
| Interaction logs | 1,983 query–response pairs | 43.1% code-mixed, 34.9% English, 22.0% Romanized Nepali |
| Satisfaction | 17 items × 26 students | CHISM-derived, four sub-scales |

## 10. Training procedure

There is **no end-to-end model training**. The encoder, cross-encoder and LLM are used
off-the-shelf; the only trained component is the character n-gram code-switch classifier.
The "training" that matters is corpus construction, index building and prompt iteration
(three internal pilots with eight volunteers).

## 11. Evaluation metrics

* Retrieval: P@5, MRR, R@10 on 300 annotated queries.
* Generation: BERTScore F1 + human ratings (correctness, pedagogical clarity, linguistic
  appropriateness), ICC = 0.79.
* Learning: gain-score t-test with Cohen's d, ANCOVA with pre-test covariate, 95% CIs,
  Shapiro–Wilk and Levene assumption checks, power analysis (G*Power).
* Perception: 17-item Likert instrument, four sub-scales.
* Engagement: message counts, code-mixing ratios, topic distributions, latency.

## 12. Results reported

| Quantity | Value |
|---|---|
| P@5 hybrid — English / code-mixed / Romanized Nepali | 0.83 / 0.79 / 0.71 |
| Dense-only P@5 (Romanized Nepali) | 0.59 → 0.71 with hybrid |
| BERTScore F1 overall | 0.872 |
| Learning gain (exp vs ctrl) | 26.2 vs 14.5 pp |
| Between-group gain difference | 11.7 pp, 95% CI 6.2–17.2 |
| t(50), d | 4.21, p < 0.001, d = 1.17 |
| ANCOVA adjusted post-test difference | 12.1 pp, 95% CI 7.4–16.8 |
| Satisfaction overall | 4.31 / 5 |
| Code-mixed share of logged queries | 43.1% |
| Engagement–gain correlation | r = 0.43, p = 0.028 |

## 13. Limitations

Acknowledged by the authors: single institution, single instructor, n = 52; section-level
rather than individual randomization; generation evaluation limited to BERTScore plus
human ratings; no long-term retention measurement; Hawthorne/novelty/instructor-as-
evaluator confounds inflating the effect size.

Not acknowledged, but visible in the text:

1. **The instructor built the system, taught both sections and set the tests.** This is the
   single largest threat to validity and outweighs the statistical machinery reported.
2. **Internal inconsistencies.**
   - Table 6 gain SDs (8.1 / 9.7) imply d = 1.31 and t = 4.72, not the reported 1.17 / 4.21
     (which imply a pooled SD of exactly 10.0).
   - Section 6.6 reports 38.1 messages per student, but 1,983 pairs over 26 students is 76.3.
   - Figure 2 says 1,247 chunks; the text and Table 1 say 1,252.
   - Table numbering is scrambled (Section 6.2 refers to "Table 7" for assumption checks,
     which is actually Table 5); Section 6.4 is missing.
   - Reference [17] is cited as "Bhandari" but appears in the list as "Rimal & Rimal".
   - References are unnumbered while the text uses numeric citations.
3. **Post-hoc power.** Power computed at the *observed* d = 1.17 is a monotone function of
   the p-value and carries no evidential weight; the reported a priori analysis used d = 1.00.
4. **P@5 with roughly ten relevant chunks per concept** is a weak ceiling test; nDCG or
   graded relevance would be more informative.
5. **No baseline chatbot condition.** The comparison is chatbot vs no chatbot, so the
   contribution of *code-mixing awareness* is never isolated from the contribution of
   having any tutor at all. A third arm (English-only RAG chatbot) is the missing cell.
6. Six weeks of access inside a 12-week measurement window means the gain covers a period
   in which the tool was absent for half the time.

## 14. Possible improvements

1. **Three-arm design**: no chatbot / English-only RAG / code-mix-aware RAG. This is the
   experiment that would actually test the paper's thesis.
2. Multi-institution, multi-instructor, individually randomized replication with a blinded
   assessor and pre-registration.
3. Fine-tune the encoder with a contrastive objective (InfoNCE / MultipleNegativesRanking)
   on (Romanized Nepali query, English chunk) pairs mined from the logs — the single
   cheapest way to lift the 0.71 Romanized-Nepali P@5.
4. Report faithfulness/attribution metrics (RAGAS-style: answer-context entailment) rather
   than BERTScore alone, since BERTScore is insensitive to fabricated specifics.
5. Add graded relevance and nDCG; annotate near-miss vs topical-drift failures separately.
6. Measure delayed retention (4–8 weeks post-test) and transfer to unseen topics.
7. Release the code-mixed corpus and the 300-query benchmark; the field has no shared
   Romanized-Nepali educational retrieval benchmark, and that artefact would outlast the
   study itself.
8. Log-level analysis of *which* questions become askable in L1 — the paper's most
   original claim currently rests on twelve interviews.
