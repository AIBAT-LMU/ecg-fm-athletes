# ECG foundation models for cardiovascular screening in athletes

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.24478226-blue)](https://doi.org/10.5281/zenodo.24478226)

Code for the paper

> Goelz C, Lang S, Schlichtiger J, Brunner S, Vieluf S. **ECG foundation models for cardiovascular
> screening in a cohort of young competitive athletes.** *npj Digital Medicine* 9, 920 (2026).
> https://doi.org/10.1038/s41746-026-03474-9

Three ECG foundation models (ECG-FM, HuBERT-ECG, ECGFounder) are fine-tuned on the
PhysioNet/CinC Challenge 2021 data and evaluated on its test split, on CODE-15% and on
expert-annotated ECGs of 43 competitive athletes. This repository contains the preprocessing,
fine-tuning, evaluation and statistical analysis, and reproduces every table and data-driven figure
of the paper.

## Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Data](#data)
- [Models](#models)
- [Reproducing the results](#reproducing-the-results)
- [What to expect when reproducing](#what-to-expect-when-reproducing)
- [Repository layout](#repository-layout)
- [Development](#development)
- [License and citation](#license-and-citation)

## Requirements

- Linux, Python 3.10
- [uv](https://docs.astral.sh/uv/) for the Python environment (recommended; pip works as well)
- An NVIDIA GPU for fine-tuning and evaluation. The results were produced with PyTorch 2.8.0
  (CUDA 12.8) on an NVIDIA A100 (40 GB). Statistics, tables and figures run on a CPU.

All dependencies are pinned in `pyproject.toml`; `uv.lock` fixes the complete environment and
`requirements.txt` is exported from it for installation with pip.

## Installation

```bash
git clone https://github.com/AIBAT-LMU/ecg-fm-athletes
cd ecg-fm-athletes
make setup            # same as: uv sync --frozen
```

Run all commands from the repository root. `make help` lists the available targets.

### Without uv

`requirements.txt` lists the same pinned versions as `uv.lock`. With Python 3.10:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps -e .
```

The `make` targets start Python through uv by default. With the environment activated, tell
them to use it directly:

```bash
export RUN=python HF_DOWNLOAD="hf download"
make evaluate
```

The development tools for `make test` and `make lint` are not part of `requirements.txt`; install
them with `pip install pytest black flake8 isort` and run them directly, for example `pytest`.

## Data

Paths in `configs/` are relative to the repository root. Place the datasets as shown below, or
make `data/` a symbolic link to a directory on another disk.

```
data/
├── external/
│   ├── physionet2021/training/   PhysioNet/CinC Challenge 2021, training data
│   └── code15/                   CODE-15%: exams.csv and exams_part0.hdf5
└── raw/
    └── lmu_sportscar/            LMU athlete ECGs and labels (not public)
```

| Dataset | Access |
|---|---|
| PhysioNet/CinC Challenge 2021 (version 1.0.3) | Public, https://doi.org/10.13026/34va-7q14 |
| CODE-15% (version 1.0.0), first part (`exams_part0.hdf5`) | Public, https://doi.org/10.5281/zenodo.4916206 |
| LMU athlete dataset | Not public because of data protection regulations and the conditions of the ethics approval. Available from the corresponding author on reasonable request, subject to approval by the ethics committee and a data use agreement. |

## Models

**Base models** are downloaded automatically to `models/` at the revisions used in the paper:

| Model | Source | Revision |
|---|---|---|
| ECG-FM | [wanglab/ecg-fm](https://huggingface.co/wanglab/ecg-fm), `mimic_iv_ecg_physionet_pretrained.pt` | `abf3d8c` |
| HuBERT-ECG (small) | [Edoardo-BS/hubert-ecg-small](https://huggingface.co/Edoardo-BS/hubert-ecg-small) | `a6d3f2c` |
| ECGFounder | [PKUDigitalHealth/ECGFounder](https://huggingface.co/PKUDigitalHealth/ECGFounder), `12_lead_ECGFounder.pth` | `d9b1793` |

**Fine-tuned checkpoints** of the paper are available on the Hugging Face Hub:

```bash
make download-models   # writes the three files to models/finetuned/
```

| Model | Repository | License |
|---|---|---|
| ECG-FM | [AIBAT-LMU/ecg-fm-athletes-ecgfm](https://huggingface.co/AIBAT-LMU/ecg-fm-athletes-ecgfm) | MIT |
| HuBERT-ECG (small) | [AIBAT-LMU/ecg-fm-athletes-hubert-ecg-small](https://huggingface.co/AIBAT-LMU/ecg-fm-athletes-hubert-ecg-small) | CC BY-NC 4.0 |
| ECGFounder | [AIBAT-LMU/ecg-fm-athletes-ecgfounder](https://huggingface.co/AIBAT-LMU/ecg-fm-athletes-ecgfounder) | MIT |

## Reproducing the results

The pipeline consists of numbered scripts. Each step has a `make` target.

| Step | Script | `make` target | Output |
|---|---|---|---|
| 1 | `01_preprocess.py` | `preprocess` | `data/processed/` |
| 2 | `02_finetune.py` | `finetune` | `runs/` (optional, see below) |
| 3 | `03_evaluate.py` | `evaluate-physionet2021`, `evaluate-code15` | `results/<model>_<dataset>/` |
| 4 | `04_evaluate_athletes.py` | `evaluate-lmu` | `results/<model>_lmu/` |
| 5 | `05_statistics.py` | `statistics` | Tables 3–7 |
| 6 | `06_bootstrap.py` | `bootstrap` | Supplementary Tables 6–10, Supplementary Fig. 9 |
| 7 | `07_figures.py` | `figures` | Figs. 1–3, the plots embedded in Fig. 4, supplementary figures |
| 8 | `08_tables.py` | `tables` | Tables 1–2, per-label results on CODE-15% |
| 9 | `09_check_reproduction.py` | `check` | Comparison with the published values |

With the published checkpoints:

```bash
make download-models   # fine-tuned checkpoints
make preprocess        # all datasets, for each model
make evaluate          # all models on all datasets
make reports           # steps 5-8: tables to reports/tables/, figures to reports/figures/
make check             # compares the tables with the values published in the paper
```

Single runs follow the pattern `make <step>-<dataset>-<model>`, for example
`make evaluate-code15-founder`. Datasets are `physionet2021`, `code15` and `lmu`; models are
`ecgfm`, `hubert` and `founder`.

`make` starts Python with `uv run --frozen --no-sync python`. To use an already activated
environment instead, for example inside a batch job, add `RUN=python`.

**Without the athlete data** you can run the preprocessing and evaluation for PhysioNet/CinC 2021
and CODE-15% (`make preprocess-physionet2021 preprocess-code15`, then
`make evaluate-physionet2021 evaluate-code15`). The statistics, most figures and the
PhysioNet rows of Table 2 compare against the athlete cohort and need the LMU data.

**Fine-tuning** (`make finetune`) reproduces the training procedure of the paper but not the exact
weights: the training script does not fix the random seed. Use the published checkpoints to
reproduce the reported numbers. To evaluate your own checkpoints, pass their paths, for example
`make evaluate CKPT_ecgfm=runs/finetune_ecgfm_prod/checkpoints/best_model_ecgfm.pth`.

### Runtime

Measured on one NVIDIA A100:

| Step | Duration |
|---|---|
| Evaluation on PhysioNet/CinC 2021 (training, validation and test split) | ECG-FM about 2 h, HuBERT-ECG about 1 h 10 min, ECGFounder about 45 min |
| Evaluation on CODE-15% | 10–20 min per model |
| Evaluation on the athlete data | 1 min per model |
| Statistics, tables, figures (CPU) | about 1 min each |
| Athlete-level bootstrap (CPU) | about 10 min |

## What to expect when reproducing

- **Main tables.** With the published checkpoints, `make check` confirms 136 values of
  Tables 1, 2, 3, 5, 6 and 7 at the precision reported in the paper. One permutation p-value of a
  clearly non-significant comparison (Table 7, HuBERT-ECG, AUPRC) comes out as 0.727 instead of
  0.726; the check allows a tolerance of 0.002 for these p-values.
- **Predictions are not bit-identical across software and hardware.** Segment-level probabilities
  of the same checkpoint can differ by up to about 0.01 between environments, typically much less. Aggregated
  metrics are unaffected at the reported precision. In the supplementary athlete-level analyses,
  individual values can differ by one unit in the last reported digit; no conclusion changes.
- **Macro averages.** `03_evaluate.py` prints a macro average over all 26 model labels. Table 2
  averages over the labels that are evaluable in the athlete cohort (PhysioNet/CinC 2021) or
  available in CODE-15% (six labels). The printed value for CODE-15% is therefore much lower than
  the value in Table 2, which is produced by `08_tables.py`.
- **Figures.** The content is reproduced exactly. The layout can differ by a few pixels depending
  on the matplotlib version and the installed fonts. Figures 4 and 5 are schematic drawings; only
  the two plots embedded in Fig. 4 are generated by code.

## Repository layout

```
├── configs/                 YAML configurations for preprocessing, fine-tuning and evaluation
├── scripts/                 numbered pipeline steps (01-09)
├── src/ecgath/              Python package
│   ├── analysis.py          shared statistics (bootstrap, permutation test, model comparison)
│   ├── data/                datasets and data loaders
│   ├── metrics/             evaluation metrics
│   ├── models/              model definitions and loading
│   ├── preprocessing/       signal processing, metadata, splits
│   └── training/            training and evaluation loops
├── tests/                   unit tests and the published reference values
├── huggingface/             model cards and upload script of the published checkpoints
├── Makefile
├── pyproject.toml
├── uv.lock
└── requirements.txt        exported from uv.lock
```

Not part of the repository and created locally: `data/`, `models/`, `results/`, `reports/`,
`runs/` and `mlruns/` (fine-tuning is tracked with MLflow).

## Development

```bash
make test     # unit tests of the analysis functions, no data needed
make lint     # flake8, isort, black
```

## License and citation

The code is released under the MIT license (see `LICENSE`).
`src/ecgath/models/third_party/ecg_founder.py` is adapted from the
[ECGFounder](https://github.com/PKUDigitalHealth/ECGFounder) repository (MIT license). The
fine-tuned checkpoints carry the licenses of their base models, listed above.

If you use this code or the checkpoints, please cite the paper (see also `CITATION.cff`):

```bibtex
@article{goelz2026ecg,
  title   = {ECG foundation models for cardiovascular screening in a cohort of young competitive athletes},
  author  = {Goelz, Christian and Lang, Simon and Schlichtiger, Jenny and Brunner, Stefan and Vieluf, Solveig},
  journal = {npj Digital Medicine},
  volume  = {9},
  pages   = {920},
  year    = {2026},
  doi     = {10.1038/s41746-026-03474-9}
}
```

Please also cite the base models and the datasets you use.

## Contact

Christian Goelz, LMU University Hospital, LMU Munich — Christian.Goelz@med.uni-muenchen.de
