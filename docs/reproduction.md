# Reproduction guide

## 0. Environment

```bash
git clone <your-fork> algosathi && cd algosathi
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Optional, for the paper-faithful stack (≈4 GB of downloads, GPU recommended):

```bash
pip install -r requirements-full.txt
# local LLM:
ollama pull llama3.1:8b-instruct-q4_K_M && ollama serve
# or hosted fallback:
export OPENAI_API_KEY=sk-...
```

Nothing else changes: the code detects what is installed and logs which backend it used.

## 1. One command

```bash
make all          # data → index → train → evaluate → results/REPORT.md
```

Equivalent to:

```bash
python -m experiments.run_all --config configs/default.yaml
```

Runtime on a 4-core laptop, offline backends: ~4 minutes (dominated by SVD fitting).

## 2. Stage by stage

```bash
# 1. Data
python -m src.data.build_knowledge_base --config configs/default.yaml
python -m src.data.build_eval_sets      --config configs/default.yaml
python -m src.data.simulate_study_data  --config configs/default.yaml

# 2. Index (offline phase of Figure 2)
python -m src.retrieval.index_builder --config configs/default.yaml

# 3. Train the code-switch classifier
python -m src.training.train_lid --config configs/default.yaml --epochs 40

# 4. Evaluate
python -m src.evaluation.evaluate_retrieval  --config configs/ablation.yaml
python -m src.evaluation.evaluate_generation --config configs/default.yaml --n 40
python -m src.evaluation.evaluate_learning   --config configs/default.yaml

# 5. Experiments
python -m experiments.hyperparameter_search --param rrf_k --values 10,30,60,100
python -m experiments.hyperparameter_search --config configs/hparams.yaml

# 6. Use it
python -m src.inference.cli --config configs/default.yaml
streamlit run streamlit_app/app.py

# 7. Tests
pytest tests -q
```

## 3. Expected artefacts

```
data/processed/    knowledge_base.jsonl (1,252)  eval_queries.jsonl (300)
                   lid_dataset.jsonl (4,000)     study_scores.csv (52)
                   interaction_logs.csv (1,983)  satisfaction_responses.csv (442)
checkpoints/       lid_model.pkl  index/{encoder,vector_store,bm25}.pkl  index/chunks.jsonl
results/tables/    table3_retrieval.csv  table4_generation.csv
                   table5_assumptions.csv  table6_learning_outcomes.csv
                   table7_satisfaction.csv  retrieval_per_query.csv  hparam_sweep.csv
results/figures/   lid_training_curves.png  retrieval_ablation_p5.png
                   learning_outcomes.png  satisfaction.png  query_mix.png  topic_gains.png
results/REPORT.md  side-by-side comparison with the paper
```

## 4. Running on your own data

**Course material**

```bash
cp my_notes/*.md data/raw/
python -m src.data.ingest --input data/raw --topic "Sorting Algorithms" --language en
python -m src.retrieval.index_builder --config configs/default.yaml
```

**Real classroom scores** — a CSV with `student_id, group, pre_test, post_test`
(`group` ∈ {experimental, control}):

```bash
python -m src.evaluation.evaluate_learning --scores path/to/scores.csv
```

