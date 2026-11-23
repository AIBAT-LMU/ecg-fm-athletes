"""Shared building blocks of the statistical analysis scripts (05_statistics.py, 06_bootstrap.py).

Keeps the analysis parameters, the input file layout and the label handling in one place so that
all scripts work on exactly the same cohort and use the same resampling procedure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)

# Analysis parameters used throughout the paper
RNG_SEED = 42
N_BOOT = 5000  # bootstrap iterations for confidence intervals
N_PERM = 5000  # permutations for permutation tests
ALPHA = 0.05

# Model keys as used in the result directory names (<model>_<dataset>/)
MODELS: List[str] = ["ecgfm", "founder", "hubert"]

DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_LABELS_CSV = Path("data/raw/lmu_sportscar/ECG_Data_Labeled.csv")
DEFAULT_TABLES_DIR = Path("reports/tables")
DEFAULT_FIGURES_DIR = Path("reports/figures")

# LMU label columns that are merged to match the PhysioNet/CinC 2021 label set
COLUMN_MERGES: Dict[str, List[str]] = {
    "CLBBB|LBBB": ["CLBBB", "LBBB"],
    "CRBBB|RBBB": ["CRBBB", "RBBB"],
    "PAC|SVPB": ["PAC", "SVPB"],
    "PVC|VPB": ["PVC", "VPB"],
}


@dataclass(frozen=True)
class ResultPaths:
    """Locations of the evaluation outputs of one model."""

    raw_segments: Path  # LMU segment-level probabilities (04_evaluate_athletes.py)
    agg_report: Path  # LMU per-label metrics after aggregation per athlete
    physionet_report: (
        Path  # per-label metrics on the PhysioNet/CinC 2021 test split (03_evaluate.py)
    )

    @classmethod
    def for_model(cls, results_dir: Path, model: str) -> "ResultPaths":
        return cls(
            raw_segments=results_dir / f"{model}_lmu" / "raw_segment_predictions.csv",
            agg_report=results_dir / f"{model}_lmu" / "athlete_aggregated_metrics_report.csv",
            physionet_report=results_dir
            / f"{model}_physionet2021"
            / "evaluation_results_test.csv",
        )

    def missing(self) -> List[Path]:
        return [
            p
            for p in (self.raw_segments, self.agg_report, self.physionet_report)
            if not p.exists()
        ]


def result_paths(results_dir: Path) -> Dict[str, ResultPaths]:
    """Result locations of all models."""
    return {m: ResultPaths.for_model(results_dir, m) for m in MODELS}


def check_inputs(labels_csv: Path, paths: Dict[str, ResultPaths]) -> None:
    """Fail early with a list of all missing input files."""
    missing = [p for p in [labels_csv] if not p.exists()]
    for model_paths in paths.values():
        missing.extend(model_paths.missing())
    if missing:
        raise FileNotFoundError("Missing input files:\n  " + "\n  ".join(str(p) for p in missing))


def load_true_labels(labels_csv: Path, merge_columns: bool = True) -> pd.DataFrame:
    """Load the athlete-level LMU labels (Ja/Nein) as 0/1.

    Args:
        merge_columns: merge the LMU columns listed in ``COLUMN_MERGES`` to the model label set.
    """
    with pd.option_context("future.no_silent_downcasting", True):
        true_labels = pd.read_csv(labels_csv, index_col=0).replace({"Ja": 1, "Nein": 0})
    true_labels = true_labels.infer_objects()
    true_labels.index = true_labels.index.astype(str)
    if not merge_columns:
        return true_labels
    for new_col, old_cols in COLUMN_MERGES.items():
        if all(col in true_labels.columns for col in old_cols):
            true_labels[new_col] = true_labels[old_cols].any(axis=1).astype(int)
            true_labels = true_labels.drop(columns=old_cols, errors="ignore")
    return true_labels


def load_per_label_report(report_csv: Path) -> pd.DataFrame:
    """Load a per-label metrics report without its global summary row.

    The result is indexed by class and keeps ``Class`` as a column.
    """
    report = pd.read_csv(report_csv)
    report["Class"] = report["Class"].astype(str)
    per_label = report[~report["Class"].str.contains("GLOBAL", case=False)].copy()
    return per_label.reset_index(drop=True).set_index("Class", drop=False)


def athlete_id(segment_id: str) -> str:
    """Athlete id of a segment id of the form ``<athlete>_<segment index>``."""
    segment_id = str(segment_id)
    return segment_id.split("_")[0] if "_" in segment_id else segment_id


def aggregate_segments_to_athletes(raw_segment_df: pd.DataFrame) -> pd.DataFrame:
    """Average the segment probabilities per athlete."""
    df = raw_segment_df.copy()
    if df.index.name == "ecg_segment_id":
        segment_ids = df.index.astype(str)
    else:
        segment_ids = df["ecg_segment_id"].astype(str)
    df["id"] = [athlete_id(s) for s in segment_ids]
    return df.groupby("id").mean(numeric_only=True)


def bootstrap_macro(
    per_label_values: Sequence[float], n_boot: int = N_BOOT, seed: int = RNG_SEED
) -> Tuple[float, float, float]:
    """Label-level bootstrap of a macro average (resampling per-label metric values).

    Returns:
        (mean, lower, upper) of the 95% percentile interval; NaN with fewer than two valid values.
    """
    rng = np.random.RandomState(seed)
    vals = np.array(per_label_values, dtype=float)
    vals = vals[~np.isnan(vals)]
    if vals.size < 2:
        return np.nan, np.nan, np.nan
    boots = np.empty(n_boot, dtype=float)
    n = len(vals)
    for i in range(n_boot):
        boots[i] = rng.choice(vals, size=n, replace=True).mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(vals.mean()), float(lo), float(hi)


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