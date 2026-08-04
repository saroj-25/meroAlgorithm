# Findings: where this reproduction agrees with the paper, and where it does not

The knowledge base and the 52-student dataset in this repository are **synthetic**. The
statistical results reproduce because the simulator is calibrated to the paper's reported
moments; the retrieval results do **not** reproduce the paper's corpus. Read what follows
as "does the described method behave as described", not as independent confirmation.

## 1. What reproduces exactly

| Quantity | Paper | Here |
|---|---|---|
| Knowledge base composition | 1,252 chunks (1,068 EN + 184 CM), per-topic | identical by construction |
| Gain difference | 11.7 pp (CI 6.2–17.2) | 11.7 pp (CI 6.1–17.3) |
| t(50), Cohen's d | 4.21, 1.17 | 4.22, 1.17 |
| ANCOVA adjusted difference | 12.1 (CI 7.4–16.8) | 12.0 (CI 7.0–17.0) |
| Assumption checks | all satisfied | all satisfied |
| Code-mixed share of logs | 43.1% | 43.1% |

The ANCOVA CI is slightly wider here because the simulator cannot also match the paper's
unreported pre/post correlation exactly while matching every marginal moment.

## 2. Two arithmetic inconsistencies in the paper

**(a) Gain SDs vs the test statistic.** Table 6 reports gain SDs of 8.1 and 9.7. Pooled,
that is 8.94, so a difference of 11.7 pp gives d = 1.31 and t(50) = 4.72 — not the
reported d = 1.17 / t = 4.21, which require a pooled SD of exactly 10.0. Both cannot be
right. `study.calibrate_to` lets you generate either version:

```yaml
study:
  calibrate_to: test_statistics   # honours d = 1.17 and t = 4.21 (default)
  # calibrate_to: reported_sds    # honours the Table 6 SDs; yields d = 1.31
```

**(b) Message counts.** Section 6.6 reports 38.1 messages per student; Section 5.3 reports
1,983 logged pairs for 26 students, i.e. 76.3 each. The simulator generates 1,983 pairs
and records both numbers in `study_summary.json`.

Also noted: Figure 2 says 1,247 chunks where the text says 1,252; Section 6.2 refers to
"Table 7" for what is printed as Table 5; Section 6.4 is missing; reference [17] is cited
as *Bhandari* but listed as *Rimal & Rimal*.

## 3. Post-hoc power

The paper reports power = 0.97 at "d = 1.17, n = 52" — the *observed* effect. Observed
power is a monotone transform of the p-value and adds nothing. `evaluate_learning.py`
therefore reports three separate numbers, clearly labelled: a priori power at the
pre-registered d = 1.00 (0.94 here), post-hoc power at the observed d (0.99), and the n
required for 80% power at d = 1.00 (17 per group).

## 4. Where the retrieval results diverge — and what it teaches

In the offline configuration (character n-gram TF-IDF+SVD encoder, synthetic corpus),
**hybrid fusion is worse than dense retrieval alone** on non-English queries:

| Retriever | English P@5 | Romanized Nepali P@5 | Code-mixed P@5 |
|---|---|---|---|
| dense only | 0.91 | 0.85 | 0.84 |
| BM25 only | 0.84 | 0.51 | 0.49 |
| hybrid + RRF, no rerank | 0.89 | 0.64 | 0.60 |
| hybrid + RRF + rerank | 0.91 | 0.74 | 0.68 |
| hybrid, weighted RRF (w_bm25 = 0.4) + rerank | 0.91 | 0.78 | 0.76 |

The paper reports the opposite ordering (hybrid lifts Romanized Nepali from 0.59 to 0.71).
Both can be true, and the reason is instructive:

* Unweighted RRF gives a document ranked #1 by *either* retriever the same credit. When
  one retriever is far weaker on a query class — BM25 on Romanized Nepali, which shares
  almost no surface tokens with English chunks — fusion imports its errors.
* In the paper, dense P@5 on Romanized Nepali is 0.59, i.e. *also* weak, so fusion is
  averaging two mediocre-but-decorrelated rankings, which is exactly when RRF wins.
* The sweep in `results/tables/hparam_sweep.csv` shows P@5 decreasing monotonically in the
  BM25 fusion weight for this corpus, which is the diagnostic to run before adopting
  hybrid retrieval anywhere.

**Practical rule this yields:** hybrid retrieval helps when the two retrievers are of
comparable strength on the query class you care about. Measure per query class before
fusing, and prefer weighted RRF when they are not.

Two further caveats on the retrieval numbers here: the synthetic queries are generated
from the same concept aliases that appear in the chunks (lexically easier than real
learner questions), and relevance is concept-level, so ~10 of 1,252 chunks are relevant
per query. Absolute values are therefore optimistic; the *ordering* across configurations
is what the ablation is for.

## 5. Generation without an LLM

The offline extractive generator scores `citation_rate = 1.00` and `grounding_rate ≈ 0.62`
by construction (it can only re-use retrieved sentences). Register match — the metric that
matters for the paper's thesis — lands at 0.92 English, 0.88 code-mixed, 0.68 Romanized
Nepali. That ordering mirrors the paper's own human ratings for linguistic
appropriateness (4.62 / 4.31 / 3.94): **pure Romanized Nepali is the hardest register to
answer in**, because the technical vocabulary that grounds the answer only exists in
English. Plugging in Llama-3.1-8B via Ollama raises fluency but does not by itself remove
this asymmetry.

## 6. One addition to the paper's pipeline

`retrieval.register_boost` (default 0.05) gives instructor-authored code-mixed chunks a
small bonus when the query itself is code-mixed, and the template generator prefers them
when replying in a Nepali register. This is our addition, not the paper's; ablate it with

```bash
python -m experiments.hyperparameter_search --param register_boost --values 0,0.05,0.15,0.3
```

## 7. What this reproduction cannot tell you

Nothing here speaks to whether AlgoSathi improves real learning. The study's dominant
threat to validity is that one person built the system, taught both sections and wrote the
tests — no amount of ANCOVA repairs that. The experiment that would test the paper's
actual thesis is a three-arm design (no chatbot / English-only RAG / code-mix-aware RAG)
with an independent instructor, and this repository is set up to run the two chatbot arms:
set `generation.preserve_register: false` and `retrieval.register_boost: 0` for the
English-only arm.
