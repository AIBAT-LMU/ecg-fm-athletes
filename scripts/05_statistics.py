#!/usr/bin/env python
"""Label-level statistics on the LMU athlete cohort.

Produces the numbers behind Tables 3-7 of the paper:

* Table 3 - macro metrics on the LMU data with label-level bootstrap confidence intervals and the
  test against the chance-level baseline (``validity_summary_extended.csv``)
* Table 4 - per-label metrics of all models on the LMU data (``per_label_lmu_metrics.csv``)
* Table 5 - Friedman test for differences between the models
  (``model_comparison_friedman_summary.csv``)
* Table 6 - post-hoc Wilcoxon signed-rank tests with Benjamini-Hochberg correction
  (``model_comparison_posthoc_<metric>.csv``, written for metrics with a significant Friedman test)
* Table 7 - domain shift between PhysioNet/CinC 2021 and LMU, one-sided paired permutation test
  (``transfer_permutation_results_extended.csv``)

All tests operate on per-label metric values. The athlete-level sensitivity analysis is in
``06_bootstrap.py``.

Inputs (written by ``03_evaluate.py`` and ``04_evaluate_athletes.py``)::

    <results-dir>/<model>_lmu/raw_segment_predictions.csv
    <results-dir>/<model>_lmu/athlete_aggregated_metrics_report.csv
    <results-dir>/<model>_physionet2021/evaluation_results_test.csv

Usage::

    python scripts/05_statistics.py --results-dir results
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

from ecgath.analysis import (
    ALPHA,
    DEFAULT_LABELS_CSV,
    DEFAULT_RESULTS_DIR,
    DEFAULT_TABLES_DIR,
    MODELS,
    N_BOOT,
    N_PERM,
    RNG_SEED,
    ResultPaths,
    athlete_id,
    bootstrap_macro,
    check_inputs,
    load_per_label_report,
    load_true_labels,
    result_paths,
)

logger = logging.getLogger("statistics")

NAME_MAP: Dict[str, str] = {"ecgfm": "ECG-FM", "founder": "ECGFounder", "hubert": "HuBERT-ECG"}

# Metrics that are tested, in the order that defines the bootstrap seeds (RNG_SEED + position)
TEST_METRICS: List[str] = ["AUROC", "AUPRC", "Recall", "Specificity", "F1-Score"]
# Metrics listed in the per-label table
REPORT_METRICS: List[str] = ["AUROC", "AUPRC", "F1-Score", "Precision", "Recall", "Specificity"]

# Column titles of the baseline test in the validity table
BASELINE_COLUMNS: Dict[str, Tuple[str, str]] = {
    "AUROC": ("Macro-AUROC", "Above Baseline (AUROC > 0.5)?"),
    "AUPRC": ("Macro-AUPRC", "Above Baseline (AUPRC > Prev.)?"),
    "Recall": ("Macro-Recall", "Above Baseline (Recall > 0.5)?"),
    "Specificity": ("Macro-Specificity", "Above Baseline (Spec. > 0.5)?"),
    "F1-Score": ("Macro-F1-Score", "Above Baseline (F1 > 0.5)?"),
}


# --------------------------------------------------------------------------------------
# Data containers
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Cohort:
    """Everything the tests need: labels, per-label metrics and the evaluable athletes."""

    true_labels: pd.DataFrame  # athlete x label, 0/1
    lmu_metrics: Dict[str, pd.DataFrame]  # per-label metrics on LMU, indexed by class
    physio_metrics: Dict[str, pd.DataFrame]  # per-label metrics on PhysioNet/CinC 2021
    athletes: List[str]  # athletes with labels and predictions from every model
    evaluable_labels: Dict[str, List[str]]  # labels with positive and negative athletes


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def athletes_with_predictions(raw_segments_csv: Path) -> set:
    """Athlete ids present in a segment prediction file (segment id = ``<athlete>_<index>``)."""
    raw = pd.read_csv(raw_segments_csv, index_col=0)
    return set(athlete_id(s) for s in raw.index)


def find_evaluable_labels(lmu_report: pd.DataFrame, n_athletes: int) -> List[str]:
    """Labels that have both positive and negative athletes, so ranking metrics are defined."""
    support_col = "Support" if "Support" in lmu_report.columns else "support"
    if support_col in lmu_report.columns:
        support = pd.to_numeric(lmu_report[support_col], errors="coerce").fillna(0).astype(int)
        mask = (support > 0) & (support < n_athletes)
    else:
        mask = ~lmu_report["AUROC"].isna()
    return sorted(list(lmu_report.loc[mask, "Class"]))


def load_cohort(labels_csv: Path, paths: Dict[str, ResultPaths]) -> Cohort:
    true_labels = load_true_labels(labels_csv)
    lmu_metrics = {m: load_per_label_report(paths[m].agg_report) for m in MODELS}
    physio_metrics = {m: load_per_label_report(paths[m].physionet_report) for m in MODELS}

    labelled = set(true_labels.index)
    per_model = [labelled & athletes_with_predictions(paths[m].raw_segments) for m in MODELS]
    athletes = sorted(set.intersection(*per_model))
    logger.info("Athletes used: %d of %d with labels", len(athletes), len(labelled))

    evaluable = {m: find_evaluable_labels(lmu_metrics[m], len(athletes)) for m in MODELS}
    for m in MODELS:
        logger.info("%s: %d evaluable labels", NAME_MAP[m], len(evaluable[m]))
    return Cohort(true_labels, lmu_metrics, physio_metrics, athletes, evaluable)


# --------------------------------------------------------------------------------------
# Statistical procedures
# --------------------------------------------------------------------------------------
def paired_permutation_test(
    x: Sequence[float],
    y: Sequence[float],
    n_perm: int = N_PERM,
    seed: int = RNG_SEED,
    alternative: str = "two-sided",
) -> Dict[str, float]:
    """Paired permutation test on the mean difference ``x - y``.

    Each permutation swaps the two values of a random subset of pairs.

    Args:
        alternative: ``"two-sided"``, ``"greater"`` (x > y) or ``"less"`` (x < y).

    Returns:
        Number of valid pairs, observed mean difference and p-value.
    """
    rng = np.random.RandomState(seed)
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n == 0:
        return {"n_pairs": 0, "d_obs": np.nan, "p_value": np.nan}
    d_obs = float(np.mean(x - y))
    perm_stats = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        swap = rng.rand(n) < 0.5
        xp, yp = x.copy(), y.copy()
        xp[swap], yp[swap] = yp[swap], xp[swap]
        perm_stats[i] = xp.mean() - yp.mean()
    if alternative == "two-sided":
        p = (np.sum(np.abs(perm_stats) >= abs(d_obs)) + 1) / (n_perm + 1)
    elif alternative == "greater":
        p = (np.sum(perm_stats >= d_obs) + 1) / (n_perm + 1)
    else:
        p = (np.sum(perm_stats <= d_obs) + 1) / (n_perm + 1)
    return {"n_pairs": n, "d_obs": d_obs, "p_value": float(p)}


# --------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------
def per_label_table(lmu_metrics: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-label metrics of all models on the LMU data (Table 4)."""
    all_labels = sorted(list(set.union(*[set(df.index) for df in lmu_metrics.values()])))
    rows = []
    for label in all_labels:
        row = {"Label": label}
        for m in MODELS:
            if label in lmu_metrics[m].index:
                for metric in REPORT_METRICS:
                    row[f"{NAME_MAP[m]}_{metric}"] = lmu_metrics[m].loc[label, metric]
        rows.append(row)
    return pd.DataFrame(rows)


def validity_table(cohort: Cohort, n_boot: int = N_BOOT) -> pd.DataFrame:
    """Macro metrics with label-level bootstrap intervals and the baseline test (Table 3).

    A metric is above baseline if the lower interval limit exceeds 0.5, or for AUPRC the mean
    label prevalence over the evaluable labels.
    """
    rows = []
    for m in MODELS:
        report = cohort.lmu_metrics[m]

        labels = [lbl for lbl in cohort.evaluable_labels[m] if lbl in cohort.true_labels.columns]
        if labels:
            mean_prevalence = cohort.true_labels.loc[cohort.athletes, labels].mean(axis=1).mean()
        else:
            mean_prevalence = np.nan
        baselines = {metric: 0.5 for metric in TEST_METRICS}
        baselines["AUPRC"] = mean_prevalence

        row: Dict[str, object] = {"Model": m}
        for i, metric in enumerate(TEST_METRICS):
            values = pd.to_numeric(report[metric], errors="coerce").values
            mean, lo, hi = bootstrap_macro(values, n_boot=n_boot, seed=RNG_SEED + i)
            above = (
                (lo > baselines[metric])
                if not math.isnan(lo) and not math.isnan(baselines[metric])
                else False
            )
            estimate_col, baseline_col = BASELINE_COLUMNS[metric]
            row[estimate_col] = f"{mean:.2f} ({lo:.2f}–{hi:.2f})"
            row[baseline_col] = above
        rows.append(row)
    return pd.DataFrame(rows)


def domain_shift_table(cohort: Cohort, n_perm: int = N_PERM) -> pd.DataFrame:
    """Domain shift between PhysioNet/CinC 2021 and LMU (Table 7).

    For every model and metric the per-label values of both datasets are paired on the common
    labels and tested with a one-sided paired permutation test (PhysioNet > LMU). ``n_labels`` is
    the number of common labels; pairs with an undefined metric in either dataset are dropped
    before the test.
    """
    rows = []
    for m in MODELS:
        physio = cohort.physio_metrics[m]
        lmu = cohort.lmu_metrics[m]
        common_labels = sorted(
            list((set(physio.index.astype(str)) & set(lmu.index.astype(str))) - {"GLOBAL"})
        )
        if len(common_labels) < 2:
            logger.warning(
                "Not enough common labels for %s; skipping the domain-shift test.", NAME_MAP[m]
            )
            continue

        row: Dict[str, object] = {"model": m, "n_labels": len(common_labels)}
        for metric in TEST_METRICS:
            physio_vals = pd.to_numeric(physio.loc[common_labels, metric], errors="coerce").values
            lmu_vals = pd.to_numeric(lmu.loc[common_labels, metric], errors="coerce").values
            valid = ~np.isnan(physio_vals) & ~np.isnan(lmu_vals)
            if np.sum(valid) < 2:
                logger.warning("%s, %s: fewer than two valid label pairs.", NAME_MAP[m], metric)
                row[f"{metric}_mean_diff"] = np.nan
                row[f"{metric}_p_value"] = np.nan
                continue
            result = paired_permutation_test(
                physio_vals[valid], lmu_vals[valid], n_perm=n_perm, alternative="greater"
            )
            row[f"{metric}_mean_diff"] = result["d_obs"]
            row[f"{metric}_p_value"] = result["p_value"]
            logger.info(
                "%s, %s: mean diff = %.3f, p = %.3g (n = %d pairs)",
                NAME_MAP[m],
                metric,
                result["d_obs"],
                result["p_value"],
                result["n_pairs"],
            )
        rows.append(row)
    return pd.DataFrame(rows)


def compare_models(
    lmu_metrics: Dict[str, pd.DataFrame], metric: str
) -> Tuple[Dict[str, object], Optional[pd.DataFrame]]:
    """Friedman test across the models for one metric, with post-hoc tests if significant.

    Labels with an undefined metric for any model are excluded. Post-hoc comparisons are two-sided
    Wilcoxon signed-rank tests, corrected with the Benjamini-Hochberg procedure.

    Returns:
        The Friedman result and the post-hoc table (None if the Friedman test is not significant).
    """
    common_labels = sorted(
        list(set.intersection(*[set(lmu_metrics[m].index) for m in MODELS]) - {"GLOBAL"})
    )
    stacked = np.vstack(
        [
            pd.to_numeric(lmu_metrics[m].loc[common_labels, metric], errors="coerce").values
            for m in MODELS
        ]
    )
    valid = ~np.isnan(stacked).any(axis=0)
    values = stacked[:, valid]
    labels_used = [common_labels[i] for i, ok in enumerate(valid) if ok]

    if values.shape[1] < 2:
        logger.warning("%s: fewer than two valid labels; skipping the Friedman test.", metric)
        return {
            "friedman_p": np.nan,
            "n_labels_used": values.shape[1],
            "labels_used": labels_used,
        }, None

    try:
        stat, p = friedmanchisquare(*[values[i, :] for i in range(values.shape[0])])
        logger.info(
            "Friedman test (%s): chi2(%d) = %.2f, p = %.4g (n = %d labels)",
            metric,
            values.shape[0] - 1,
            stat,
            p,
            len(labels_used),
        )
    except ValueError as err:
        logger.warning("Friedman test failed for %s: %s", metric, err)
        p = np.nan

    friedman = {"friedman_p": p, "n_labels_used": len(labels_used), "labels_used": labels_used}
    if not p < ALPHA:
        return friedman, None

    pairs, p_raw = [], []
    for i in range(values.shape[0]):
        for j in range(i + 1, values.shape[0]):
            try:
                if np.allclose(values[i, :], values[j, :]):
                    p_pair = 1.0
                else:
                    _, p_pair = wilcoxon(
                        values[i, :], values[j, :], alternative="two-sided", zero_method="wilcox"
                    )
            except ValueError as err:
                logger.warning("Wilcoxon test failed for %s vs %s: %s", MODELS[i], MODELS[j], err)
                p_pair = np.nan
            pairs.append(f"{MODELS[i]} vs {MODELS[j]}")
            p_raw.append(p_pair if not np.isnan(p_pair) else 1.0)

    reject, p_adj, _, _ = multipletests(p_raw, alpha=ALPHA, method="fdr_bh")
    posthoc = pd.DataFrame(
        [
            {"pair": pair, "p_raw": p_raw[k], "p_adj_fdr": p_adj[k], "significant_fdr": reject[k]}
            for k, pair in enumerate(pairs)
        ]
    )
    return friedman, posthoc


# --------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------
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
        "--labels", type=Path, default=DEFAULT_LABELS_CSV, help="athlete-level LMU labels"
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=DEFAULT_TABLES_DIR,
        help="output directory for CSV tables (default: reports/tables)",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=N_BOOT,
        help=f"bootstrap iterations (default: {N_BOOT}, as in the paper)",
    )
    parser.add_argument(
        "--n-perm",
        type=int,
        default=N_PERM,
        help=f"permutations (default: {N_PERM}, as in the paper)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    paths = result_paths(args.results_dir)
    check_inputs(args.labels, paths)
    args.tables_dir.mkdir(parents=True, exist_ok=True)

    cohort = load_cohort(args.labels, paths)

    # Table 4: per-label metrics
    per_label = per_label_table(cohort.lmu_metrics)
    per_label.to_csv(args.tables_dir / "per_label_lmu_metrics.csv", index=False)

    # Table 3: macro metrics against the chance-level baseline
    validity = validity_table(cohort, n_boot=args.n_boot)
    validity.to_csv(args.tables_dir / "validity_summary_extended.csv", index=False)
    logger.info(
        "Macro metrics on LMU (label-level bootstrap):\n%s", validity.to_string(index=False)
    )

    # Table 7: domain shift
    shift = domain_shift_table(cohort, n_perm=args.n_perm)
    shift.to_csv(args.tables_dir / "transfer_permutation_results_extended.csv", index=False)

    # Tables 5 and 6: model comparison
    friedman_results = {}
    for metric in TEST_METRICS:
        friedman_results[metric], posthoc = compare_models(cohort.lmu_metrics, metric)
        if posthoc is not None:
            posthoc.to_csv(
                args.tables_dir / f"model_comparison_posthoc_{metric.lower()}.csv", index=False
            )
            logger.info("Post-hoc tests for %s:\n%s", metric, posthoc.to_string(index=False))
    pd.DataFrame(friedman_results).T.to_csv(
        args.tables_dir / "model_comparison_friedman_summary.csv"
    )

    logger.info("Tables written to %s", args.tables_dir)


if __name__ == "__main__":
    main()