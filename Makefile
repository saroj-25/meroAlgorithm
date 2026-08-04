# AlgoSathi - common tasks.  Run `make help` for the list.
PYTHON  ?= python
CONFIG  ?= configs/default.yaml
ABL     ?= configs/ablation.yaml

.PHONY: help install data index train retrieval generation learning experiments \
        evaluate all app cli test clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	 awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:      ## install core dependencies
	$(PYTHON) -m pip install -r requirements.txt

install-full: ## install the paper-faithful stack (torch, faiss, bert-score, mlflow)
	$(PYTHON) -m pip install -r requirements.txt -r requirements-full.txt

data:         ## build knowledge base, evaluation sets and study data
	$(PYTHON) -m src.data.build_knowledge_base --config $(CONFIG)
	$(PYTHON) -m src.data.build_eval_sets      --config $(CONFIG)
	$(PYTHON) -m src.data.simulate_study_data  --config $(CONFIG)

index:        ## encode the corpus and build the vector + BM25 indexes
	$(PYTHON) -m src.retrieval.index_builder --config $(CONFIG)

train:        ## train the code-switch language-ID classifier
	$(PYTHON) -m src.training.train_lid --config $(CONFIG)

retrieval:    ## retrieval ablation (Table 3)
	$(PYTHON) -m src.evaluation.evaluate_retrieval --config $(ABL)

generation:   ## generation quality (Table 4)
	$(PYTHON) -m src.evaluation.evaluate_generation --config $(CONFIG) --n 40

learning:     ## learning outcomes and satisfaction (Tables 5-7)
	$(PYTHON) -m src.evaluation.evaluate_learning --config $(CONFIG)

evaluate: retrieval generation learning  ## all three evaluations

experiments:  ## hyper-parameter sweep
	$(PYTHON) -m experiments.hyperparameter_search --config configs/hparams.yaml

all:          ## full reproduction + results/REPORT.md
	$(PYTHON) -m experiments.run_all --config $(CONFIG) --ablation-config $(ABL)

app:          ## launch the Streamlit demo
	streamlit run streamlit_app/app.py

cli:          ## interactive terminal chat
	$(PYTHON) -m src.inference.cli --config $(CONFIG)

test:         ## run the test suite
	$(PYTHON) -m pytest tests -q

clean:        ## remove generated artefacts (keeps code and configs)
	rm -rf data/processed/* checkpoints/index checkpoints/*.pkl \
	       results/tables/* results/figures/* results/REPORT.md \
	       experiments/runs __pycache__ .pytest_cache
	@echo "cleaned"
