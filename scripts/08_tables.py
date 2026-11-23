#!/usr/bin/env python
"""Descriptive and summary tables of the paper.

==================  ================================================================
Name                Content
==================  ================================================================
datasets            Table 1: size, age and sex distribution of the three datasets
                    (``table1_datasets.csv``)
macro_metrics       Table 2: macro metrics on the PhysioNet/CinC 2021 test split and on
                    CODE-15% (``table2_macro_metrics.csv``)
code15_per_label    Supplement: per-label metrics on CODE-15% (``code15_per_label.csv``)
==================  ================================================================

How the macro averages of Table 2 are defined:

* PhysioNet/CinC 2021 - mean over the labels for which the metric is defined on both the
  PhysioNet test split and the LMU cohort. This is the same label set as in the domain-shift
  analysis (Table 7), so Table 2 and Table 7 are directly comparable.
* CODE-15% - mean over the six labels the dataset shares with the model label set.

The macro average printed by ``03_evaluate.py`` is taken over all 26 model labels instead and is
therefore lower for CODE-15%, where 20 of them do not occur.

Only aggregated values are written; no per-person data leaves this script.

Usage::

    python scripts/08_tables.py
    python scripts/08_tables.py --only macro_metrics
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ecgath.analysis import (
    DEFAULT_RESULTS_DIR,
    DEFAULT_TABLES_DIR,
    load_per_label_report,
    result_paths,
)

logger = logging.getLogger("tables")

NAME_MAP: Dict[str, str] = {
    "ecgfm": "ECG-FM",
    "hubert": "HuBERT-ECG (small)",
    "founder": "ECGFounder",
}
# Order of the models in the tables of the paper
TABLE_MODELS: List[str] = ["ecgfm", "hubert", "founder"]

METRICS: List[str] = ["AUROC", "AUPRC", "Recall", "Specificity", "F1-Score"]
METRIC_NAMES: Dict[str, str] = {"Recall": "Sensitivity"}  # name used in the paper

DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_CODE15_EXAMS = Path("data/external/code15/exams.csv")
PHYSIONET_META_FILE = Path("physionet2021") / "physionet2021_meta.csv"
LMU_META_FILE = Path("lmu_sportscar_ecgfm") / "lmu_sportscar_meta.csv"

# CODE-15% subset used in the study and its label columns mapped to the model label set
CODE15_TRACE_FILE = "exams_part0.hdf5"
CODE15_LABELS: Dict[str, str] = {
    "AF": "AF",
    "CLBBB|LBBB": "LBBB",
    "CRBBB|RBBB": "RBBB",
    "IAVB": "1dAVb",
    "SB": "SB",
    "STach": "ST",
}

# Plausible age range; values outside are treated as missing (PhysioNet/CinC 2021)
AGE_RANGE = (1, 120)


@dataclass(frozen=True)
class Inputs:
    """Input locations given on the command line."""

    results_dir: Path
    processed_dir: Path
    code15_exams: Path
    tables_dir: Path

    def code15_report(self, model: str) -> Path:
        return self.results_dir / f"{model}_code15" / "evaluation_results_test.csv"


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    return path


def load_code15_exams(exams_csv: Path) -> pd.DataFrame:
    """Exam table of the CODE-15% subset used in the study."""
    exams = pd.read_csv(require(exams_csv))
    return exams[exams["trace_file"] == CODE15_TRACE_FILE].copy()


# --------------------------------------------------------------------------------------
# Table 1: datasets
# --------------------------------------------------------------------------------------
def describe_dataset(
    name: str, n: int, age: pd.Series, male: float, female: float, unknown: float
) -> Dict[str, object]:
    return {
        "Dataset": name,
        "Samples": n,
        "Age mean": round(age.mean(), 1),
        "Age SD": round(age.std(), 1),
        "Male (%)": male,
        "Female (%)": female,
        "Sex unknown (%)": unknown,
    }


def dataset_table(inputs: Inputs) -> pd.DataFrame:
    """Size, age and sex distribution of the datasets (Table 1)."""
    physio = pd.read_csv(require(inputs.processed_dir / PHYSIONET_META_FILE), low_memory=False)
    physio_age = pd.to_numeric(physio["Age"], errors="coerce")
    physio_age = physio_age[(physio_age >= AGE_RANGE[0]) & (physio_age <= AGE_RANGE[1])]
    sex = physio["Sex"].value_counts(normalize=True, dropna=False)

    lmu = pd.read_csv(require(inputs.processed_dir / LMU_META_FILE), low_memory=False)
    lmu_age = pd.to_numeric(lmu["age"], errors="coerce")

    code15 = load_code15_exams(inputs.code15_exams)
    code15_male = code15["is_male"].mean() * 100

    rows = [
        describe_dataset(
            "PhysioNet/CinC 2021",
            len(physio),
            physio_age,
            round(sex.get("Male", 0) * 100, 1),
            round(sex.get("Female", 0) * 100, 1),
            round(sex.get("Unknown", 0) * 100, 2),
        ),
        # The LMU cohort consists of male athletes only; the metadata holds one row per recording.
        describe_dataset("LMU", len(lmu), lmu_age, 100.0, 0.0, 0.0),
        describe_dataset(
            "CODE-15%",
            len(code15),
            code15["age"],
            round(code15_male, 1),
            round(100 - code15_male, 1),
            0.0,
        ),
    ]
    return pd.DataFrame(rows)


def make_datasets(inputs: Inputs) -> None:
    table = dataset_table(inputs)
    table.to_csv(inputs.tables_dir / "table1_datasets.csv", index=False)
    logger.info("Table 1 - datasets:\n%s", table.to_string(index=False))


# --------------------------------------------------------------------------------------
# Table 2: macro metrics on PhysioNet/CinC 2021 and CODE-15%
# --------------------------------------------------------------------------------------
def physionet_macro(physio: pd.DataFrame, lmu: pd.DataFrame) -> Dict[str, float]:
    """Macro metrics on PhysioNet over the labels that are also valid on the LMU cohort."""
    labels = sorted(list((set(physio.index) & set(lmu.index)) - {"GLOBAL"}))
    macro = {}
    for metric in METRICS:
        physio_vals = pd.to_numeric(physio.loc[labels, metric], errors="coerce").values
        lmu_vals = pd.to_numeric(lmu.loc[labels, metric], errors="coerce").values
        valid = ~np.isnan(physio_vals) & ~np.isnan(lmu_vals)
        macro[metric] = physio_vals[valid].mean()
    return macro


def code15_per_label(inputs: Inputs, model: str) -> pd.DataFrame:
    """Per-label metrics of one model on the six labels available in CODE-15%."""
    report = pd.read_csv(require(inputs.code15_report(model)))
    return report[report["Class"].isin(CODE15_LABELS)].copy()


def macro_metrics_table(inputs: Inputs) -> pd.DataFrame:
    paths = result_paths(inputs.results_dir)
    rows = []
    for model in TABLE_MODELS:
        physio = load_per_label_report(require(paths[model].physionet_report))
        lmu = load_per_label_report(require(paths[model].agg_report))
        code15 = code15_per_label(inputs, model)
        macros = {
            "PhysioNet/CinC 2021": physionet_macro(physio, lmu),
            "CODE-15%": {metric: code15[metric].mean() for metric in METRICS},
        }
        for dataset, macro in macros.items():
            row = {"Model": NAME_MAP[model], "Dataset": dataset}
            row.update({f"Macro-{METRIC_NAMES.get(m, m)}": macro[m] for m in METRICS})
            rows.append(row)
    return pd.DataFrame(rows)


def make_macro_metrics(inputs: Inputs) -> None:
    table = macro_metrics_table(inputs)
    table.to_csv(inputs.tables_dir / "table2_macro_metrics.csv", index=False)
    logger.info("Table 2 - macro metrics:\n%s", table.round(2).to_string(index=False))


# --------------------------------------------------------------------------------------
# Supplement: per-label metrics on CODE-15%
# --------------------------------------------------------------------------------------
def code15_per_label_table(inputs: Inputs) -> pd.DataFrame:
    """Per-label metrics on CODE-15% with the number of positive exams, rounded to two decimals."""
    exams = load_code15_exams(inputs.code15_exams)
    n_positive = {label: int(exams[column].sum()) for label, column in CODE15_LABELS.items()}

    parts = []
    for model in TABLE_MODELS:
        per_label = code15_per_label(inputs, model)
        per_label["Model"] = NAME_MAP[model]
        per_label["n"] = per_label["Class"].map(n_positive)
        parts.append(per_label)
    table = pd.concat(parts, ignore_index=True)[["Model", "Class", "n"] + METRICS]
    table = table.rename(columns={"Class": "Label", **METRIC_NAMES})
    metric_cols = [METRIC_NAMES.get(m, m) for m in METRICS]
    table[metric_cols] = table[metric_cols].round(2)
    return table


def make_code15_per_label(inputs: Inputs) -> None:
    table = code15_per_label_table(inputs)
    table.to_csv(inputs.tables_dir / "code15_per_label.csv", index=False)
    logger.info("Per-label metrics on CODE-15%%:\n%s", table.to_string(index=False))


# --------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------
TABLES: Dict[str, Callable[[Inputs], None]] = {
    "datasets": make_datasets,
    "macro_metrics": make_macro_metrics,
    "code15_per_label": make_code15_per_label,
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="directory with the <model>_<dataset>/ evaluation outputs (default: results)",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="directory with the preprocessed datasets (default: data/processed)",
    )
    parser.add_argument(
        "--code15-exams",
        type=Path,
        default=DEFAULT_CODE15_EXAMS,
        help="exams.csv of the CODE-15%% dataset (default: data/external/code15/exams.csv)",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=DEFAULT_TABLES_DIR,
        help="output directory (default: reports/tables)",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=list(TABLES),
        metavar="NAME",
        help="create only these tables: " + ", ".join(TABLES),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    inputs = Inputs(args.results_dir, args.processed_dir, args.code15_exams, args.tables_dir)
    inputs.tables_dir.mkdir(parents=True, exist_ok=True)
    for name in args.only or list(TABLES):
        TABLES[name](inputs)
    logger.info("Tables written to %s", inputs.tables_dir)


if __name__ == "__main__":
    main()