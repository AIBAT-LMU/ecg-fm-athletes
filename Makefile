# ECG foundation models for cardiovascular screening in athletes - reproduction pipeline
# Run all targets from the repository root. `make help` lists the available targets.

MODELS   := ecgfm hubert founder
DATASETS := physionet2021 code15 lmu

# How Python is started. --frozen/--no-sync keep the locked environment untouched.
RUN     ?= uv run --frozen --no-sync python
RESULTS ?= results

# Fine-tuned checkpoints (download from Hugging Face or produce with `make finetune`)
CKPT_ecgfm   := models/finetuned/best_model_ecgfm.pth
CKPT_hubert  := models/finetuned/best_model_hubertecgsmall.pth
CKPT_founder := models/finetuned/best_model_ecgfounder.pth

# Evaluation script per dataset: LMU labels are per athlete, so segments are aggregated first
EVAL_physionet2021 := scripts/03_evaluate.py
EVAL_code15        := scripts/03_evaluate.py
EVAL_lmu           := scripts/04_evaluate_athletes.py

.DEFAULT_GOAL := help

.PHONY: help
help:  ## Show this help
	@grep -hE '^[a-zA-Z_%-]+:.*## ' $(MAKEFILE_LIST) | sed -E 's/:.*## /\t/' | awk -F'\t' '{printf "  %-28s %s\n", $$1, $$2}'
	@echo ""
	@echo "  One dataset: make preprocess-<dataset> | evaluate-<dataset>"
	@echo "  Single runs: make preprocess-<dataset>-<model> | finetune-<model> | evaluate-<dataset>-<model>"
	@echo "  datasets: $(DATASETS)    models: $(MODELS)"

.PHONY: setup
setup:  ## Create the Python environment from uv.lock
	uv sync --frozen

.PHONY: requirements
requirements:  ## Export the locked environment to requirements.txt (for installation with pip)
	uv export --frozen --no-dev --no-emit-project --no-hashes --no-annotate -o requirements.txt

# Fine-tuned checkpoints of the paper on the Hugging Face Hub
HF_ORG ?= AIBAT-LMU
HF_DOWNLOAD ?= uv run --frozen --no-sync hf download

.PHONY: download-models
download-models:  ## Download the fine-tuned checkpoints of the paper to models/finetuned/
	$(HF_DOWNLOAD) $(HF_ORG)/ecg-fm-athletes-ecgfm $(notdir $(CKPT_ecgfm)) --local-dir models/finetuned
	$(HF_DOWNLOAD) $(HF_ORG)/ecg-fm-athletes-hubert-ecg-small $(notdir $(CKPT_hubert)) --local-dir models/finetuned
	$(HF_DOWNLOAD) $(HF_ORG)/ecg-fm-athletes-ecgfounder $(notdir $(CKPT_founder)) --local-dir models/finetuned

# ---------------------------------------------------------------------------
# 01 Preprocessing
# ---------------------------------------------------------------------------
define PREPROCESS_RULE
.PHONY: preprocess-$(1)-$(2)
preprocess-$(1)-$(2):
	$$(RUN) scripts/01_preprocess.py --config configs/preprocessing/$(1)_$(2).yml
endef
$(foreach d,$(DATASETS),$(foreach m,$(MODELS),$(eval $(call PREPROCESS_RULE,$(d),$(m)))))

define PREPROCESS_DATASET_RULE
.PHONY: preprocess-$(1)
preprocess-$(1): $(foreach m,$(MODELS),preprocess-$(1)-$(m))
endef
$(foreach d,$(DATASETS),$(eval $(call PREPROCESS_DATASET_RULE,$(d))))

.PHONY: preprocess
preprocess: $(foreach d,$(DATASETS),$(foreach m,$(MODELS),preprocess-$(d)-$(m)))  ## Preprocess all datasets for all models

# ---------------------------------------------------------------------------
# 02 Fine-tuning (PhysioNet/CinC 2021)
# ---------------------------------------------------------------------------
define FINETUNE_RULE
.PHONY: finetune-$(1)
finetune-$(1):
	$$(RUN) scripts/02_finetune.py --config configs/finetuning/$(1).yml
endef
$(foreach m,$(MODELS),$(eval $(call FINETUNE_RULE,$(m))))

.PHONY: finetune
finetune: $(foreach m,$(MODELS),finetune-$(m))  ## Fine-tune all models (not bit-reproducible, see README)

# ---------------------------------------------------------------------------
# 03/04 Evaluation
# ---------------------------------------------------------------------------
define EVALUATE_RULE
.PHONY: evaluate-$(1)-$(2)
evaluate-$(1)-$(2):
	$$(RUN) $$(EVAL_$(1)) --config configs/evaluation/$(1)_$(2).yml \
		--checkpoint $$(CKPT_$(2)) --output_dir $$(RESULTS)/$(2)_$(1)
endef
$(foreach d,$(DATASETS),$(foreach m,$(MODELS),$(eval $(call EVALUATE_RULE,$(d),$(m)))))

define EVALUATE_DATASET_RULE
.PHONY: evaluate-$(1)
evaluate-$(1): $(foreach m,$(MODELS),evaluate-$(1)-$(m))
endef
$(foreach d,$(DATASETS),$(eval $(call EVALUATE_DATASET_RULE,$(d))))

.PHONY: evaluate
evaluate: $(foreach d,$(DATASETS),$(foreach m,$(MODELS),evaluate-$(d)-$(m)))  ## Evaluate all models on all datasets (about 4.5 h on one A100)

# ---------------------------------------------------------------------------
# 05-08 Statistics, figures and tables (written to reports/)
# ---------------------------------------------------------------------------
.PHONY: statistics
statistics:  ## Label-level statistics (Tables 3-7)
	$(RUN) scripts/05_statistics.py --results-dir $(RESULTS)

.PHONY: bootstrap
bootstrap:  ## Athlete-level bootstrap (Supplementary Tables 6-10, about 10 min on CPU)
	$(RUN) scripts/06_bootstrap.py --results-dir $(RESULTS)

.PHONY: figures
figures:  ## All data-driven figures as PDF and PNG
	$(RUN) scripts/07_figures.py --results-dir $(RESULTS)

.PHONY: tables
tables:  ## Dataset description and macro metrics (Tables 1-2, CODE-15% per label)
	$(RUN) scripts/08_tables.py --results-dir $(RESULTS)

.PHONY: reports
reports: statistics bootstrap figures tables  ## All tables and figures from existing evaluation results

.PHONY: check
check:  ## Compare the generated tables with the values published in the paper
	$(RUN) scripts/09_check_reproduction.py

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------
.PHONY: test
test:  ## Unit tests of the analysis functions (no data needed, runs in seconds)
	uv run --frozen pytest

.PHONY: lint
lint:  ## Check code style
	uv run --frozen flake8 src scripts tests
	uv run --frozen isort --check --diff src scripts tests
	uv run --frozen black --check src scripts tests

.PHONY: format
format:  ## Format code
	uv run --frozen isort src scripts tests
	uv run --frozen black src scripts tests

.PHONY: clean
clean:  ## Remove Python caches
	find . -type d -name "__pycache__" -not -path "./.venv/*" -prune -exec rm -rf {} +