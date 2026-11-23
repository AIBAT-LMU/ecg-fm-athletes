#!/usr/bin/env python
"""Athlete-level bootstrap for the LMU athlete cohort.

Produces Supplementary Tables 6-10 and Supplementary Fig. 9 of the paper.

The label-level bootstrap behind Table 3 resamples per-label metric values. Because all labels
derive from the same athletes, these values are not independent. This script therefore recomputes
the 95% confidence intervals of the macro metrics by resampling athletes:

1. resample the athletes with replacement,
2. recompute every per-label metric on the resampled cohort,
3. take the macro average over the labels that are valid in the original cohort,
4. repeat ``N_BOOT`` times and take the percentile interval.

From the same bootstrap draws it derives athlete-level intervals for the domain shift
(PhysioNet/CinC 2021 vs. LMU) and for the between-model differences. It also reports the predicted
probabilities of positive cases and the NSR probability by label co-occurrence.

Inputs (written by ``03_evaluate.py`` and ``04_evaluate_athletes.py``)::

    <results-dir>/<model>_lmu/raw_segment_predictions.csv
    <results-dir>/<model>_lmu/athlete_aggregated_metrics_report.csv
    <results-dir>/<model>_physionet2021/evaluation_results_test.csv

Usage::

    python scripts/06_bootstrap.py --results-dir results
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib as mpl

mpl.use("Agg")  # no display needed; must be set before pyplot is imported

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from tqdm import tqdm  # noqa: E402

from ecgath.analysis import (  # noqa: E402
    ALPHA,
    DEFAULT_FIGURES_DIR,
    DEFAULT_LABELS_CSV,
    DEFAULT_RESULTS_DIR,
    DEFAULT_TABLES_DIR,
    MODELS,
    N_BOOT,
    RNG_SEED,
    ResultPaths,
    aggregate_segments_to_athletes,
    bootstrap_macro,
    check_inputs,
    load_per_label_report,
    load_true_labels,
    result_paths,
)

logger = logging.getLogger("bootstrap")

# --------------------------------------------------------------------------------------
# Analysis parameters (identical to the label-level analysis in 05_statistics.py)
# --------------------------------------------------------------------------------------
THRESHOLD = 0.5  # fixed decision threshold for Sensitivity / Specificity / F1

NAME_MAP: Dict[str, str] = {
    "ecgfm": "ECG-FM",
    "founder": "ECGFounder",
    "hubert": "HuBERT-ECG (small)",
}
METRICS: List[str] = ["AUROC", "AUPRC", "Recall", "Specificity", "F1-Score"]
METRIC_LABELS: Dict[str, str] = {
    "AUROC": "Macro-AUROC",
    "AUPRC": "Macro-AUPRC",
    "Recall": "Macro-Sensitivity",
    "Specificity": "Macro-Specificity",
    "F1-Score": "Macro-F1-Score",
}

# Model pairs for the between-model comparison (first minus second)
MODEL_PAIRS: List[Tuple[str, str]] = [
    ("founder", "ecgfm"),
    ("founder", "hubert"),
    ("ecgfm", "hubert"),
]

# Labels with at least nine positive athletes, used for the positive-case probabilities
POSITIVE_CASE_LABELS: List[str] = ["BBB", "Brady", "CRBBB|RBBB", "IRBBB", "RAD", "SB", "SA", "NSR"]

STYLE_CONFIG = {
    "model_colors": ["#FFDCFF", "#CD34B5", "#6D00B8"],
    "font_family": "DejaVu Sans",
    "title_fontsize": 10,
    "label_fontsize": 10,
    "tick_fontsize": 10,
    "grid_linestyle": "--",
    "grid_alpha": 0.6,
    "line_color": "#000000",
    "line_style": "--",
    "tick_direction": "out",
    "tick_width": 0.8,
    "tick_length": 4,
    "bar_linewidth": 0.8,
}


# --------------------------------------------------------------------------------------
# Data containers
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CohortData:
    """Aligned ground truth and athlete-level probabilities of one model."""

    y_true: pd.DataFrame  # (n_athletes, n_labels), 0/1
    y_prob: pd.DataFrame  # (n_athletes, n_labels), mean segment probability


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def load_cohort(true_labels: pd.DataFrame, paths: Dict[str, ResultPaths]) -> Dict[str, CohortData]:
    """Build aligned (y_true, y_prob) matrices on the athletes common to all models."""
    athlete_probs = {
        m: aggregate_segments_to_athletes(pd.read_csv(paths[m].raw_segments, index_col=0))
        for m in MODELS
    }
    final_athletes = sorted(
        set(true_labels.index).intersection(*[set(p.index) for p in athlete_probs.values()])
    )
    logger.info("Athletes used: %d", len(final_athletes))

    data: Dict[str, CohortData] = {}
    for m in MODELS:
        labels = sorted(set(true_labels.columns) & set(athlete_probs[m].columns))
        data[m] = CohortData(
            y_true=true_labels.loc[final_athletes, labels].astype(int),
            y_prob=athlete_probs[m].loc[final_athletes, labels].astype(float),
        )
        logger.info("%s: %d labels", NAME_MAP[m], len(labels))
    return data


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
def per_label_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = THRESHOLD
) -> np.ndarray:
    """Compute the metrics per label.

    Validity rules mirror the aggregated report: AUROC/AUPRC/F1 need both classes, Sensitivity
    needs positives, Specificity needs negatives.

    Args:
        y_true: (n_athletes, n_labels) binary ground truth.
        y_prob: (n_athletes, n_labels) predicted probabilities.

    Returns:
        (n_labels, len(METRICS)) array in ``METRICS`` order, NaN where a metric is undefined.
    """
    n_labels = y_true.shape[1]
    out = np.full((n_labels, len(METRICS)), np.nan)
    y_pred = (y_prob >= threshold).astype(int)
    for j in range(n_labels):
        t, p, b = y_true[:, j], y_prob[:, j], y_pred[:, j]
        n_pos, n_neg = int(t.sum()), int((1 - t).sum())
        if n_pos > 0 and n_neg > 0:
            out[j, 0] = roc_auc_score(t, p)
            out[j, 1] = average_precision_score(t, p)
        tp = int(((b == 1) & (t == 1)).sum())
        fn = int(((b == 0) & (t == 1)).sum())
        tn = int(((b == 0) & (t == 0)).sum())
        fp = int(((b == 1) & (t == 0)).sum())
        if n_pos > 0:
            out[j, 2] = tp / (tp + fn)
        if n_neg > 0:
            out[j, 3] = tn / (tn + fp)
        if n_pos > 0 and n_neg > 0:  # F1 is undefined for labels with 100% prevalence (NSR)
            out[j, 4] = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    return out


def macro_from_per_label(per_label: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Macro mean per metric over the labels that are valid in the original cohort."""
    res = np.full(len(METRICS), np.nan)
    for k in range(len(METRICS)):
        vals = per_label[valid_mask[:, k], k]
        vals = vals[~np.isnan(vals)]
        if vals.size:
            res[k] = vals.mean()
    return res


def point_estimates(
    data: Dict[str, CohortData],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Macro metrics and per-label validity masks on the full cohort."""
    point: Dict[str, np.ndarray] = {}
    valid_masks: Dict[str, np.ndarray] = {}
    for m in MODELS:
        per_label = per_label_metrics(data[m].y_true.values, data[m].y_prob.values)
        valid_masks[m] = ~np.isnan(per_label)
        point[m] = macro_from_per_label(per_label, valid_masks[m])
    return point, valid_masks


def metric_baselines(
    data: Dict[str, CohortData], valid_masks: Dict[str, np.ndarray]
) -> Dict[str, Dict[str, float]]:
    """Chance-level baselines: 0.5, except AUPRC = mean prevalence over the AUPRC-valid labels."""
    baselines = {}
    for m in MODELS:
        prevalence = data[m].y_true.values[:, valid_masks[m][:, 1]].mean()
        baselines[m] = {
            "AUROC": 0.5,
            "AUPRC": prevalence,
            "Recall": 0.5,
            "Specificity": 0.5,
            "F1-Score": 0.5,
        }
    return baselines


# --------------------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------------------
def percentile_ci(values: np.ndarray, alpha: float = ALPHA) -> Tuple[float, float]:
    """Two-sided percentile interval, ignoring NaN draws."""
    values = values[~np.isnan(values)]
    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi


def bootstrap_athletes(
    cohort: CohortData, valid_mask: np.ndarray, n_boot: int = N_BOOT, seed: int = RNG_SEED
) -> np.ndarray:
    """Resample athletes with replacement.

    Returns:
        (n_boot, len(METRICS)) array of macro metrics, one row per bootstrap draw.
    """
    rng = np.random.RandomState(seed)
    yt, yp = cohort.y_true.values, cohort.y_prob.values
    n = yt.shape[0]
    boots = np.full((n_boot, len(METRICS)), np.nan)
    for b in tqdm(range(n_boot), leave=False):
        idx = rng.randint(0, n, size=n)
        boots[b] = macro_from_per_label(per_label_metrics(yt[idx], yp[idx]), valid_mask)
    return boots


# --------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------
def athlete_ci_table(
    boot_results: Dict[str, np.ndarray],
    point: Dict[str, np.ndarray],
    valid_masks: Dict[str, np.ndarray],
    baselines: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """Athlete-level confidence intervals of the macro metrics."""
    rows = []
    for m in MODELS:
        for k, metric in enumerate(METRICS):
            lo, hi = percentile_ci(boot_results[m][:, k])
            rows.append(
                {
                    "Model": NAME_MAP[m],
                    "Metric": METRIC_LABELS[metric],
                    "n_labels": int(valid_masks[m][:, k].sum()),
                    "Estimate": point[m][k],
                    "CI_lo": lo,
                    "CI_hi": hi,
                    "Baseline": baselines[m][metric],
                    "Above baseline": bool(lo > baselines[m][metric]),
                    "Bootstrap unit": "athlete",
                }
            )
    return pd.DataFrame(rows)


def label_ci_table(
    paths: Dict[str, ResultPaths], baselines: Dict[str, Dict[str, float]], n_boot: int = N_BOOT
) -> pd.DataFrame:
    """Label-level confidence intervals (Table 3), recomputed from the aggregated reports."""
    rows = []
    for m in MODELS:
        rep = load_per_label_report(paths[m].agg_report)
        for i, metric in enumerate(METRICS):
            vals = pd.to_numeric(rep[metric], errors="coerce").values
            mean_val, lo, hi = bootstrap_macro(vals, n_boot=n_boot, seed=RNG_SEED + i)
            rows.append(
                {
                    "Model": NAME_MAP[m],
                    "Metric": METRIC_LABELS[metric],
                    "n_labels": int(np.sum(~np.isnan(vals))),
                    "Estimate": mean_val,
                    "CI_lo": lo,
                    "CI_hi": hi,
                    "Baseline": baselines[m][metric],
                    "Above baseline": bool(lo > baselines[m][metric]),
                    "Bootstrap unit": "label",
                }
            )
    return pd.DataFrame(rows)


def compare_bootstrap_units(
    label_ci: pd.DataFrame, athlete_ci: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Combine both bootstrap variants.

    Returns:
        The long table with all values and the wide table with formatted ``estimate (CI)`` strings.
    """
    comparison = pd.concat([label_ci, athlete_ci], ignore_index=True)
    comparison["CI width"] = comparison["CI_hi"] - comparison["CI_lo"]
    comparison["Estimate (95% CI)"] = comparison.apply(
        lambda r: f"{r['Estimate']:.2f} ({r['CI_lo']:.2f}–{r['CI_hi']:.2f})", axis=1
    )

    wide = comparison.pivot_table(
        index=["Model", "Metric"],
        columns="Bootstrap unit",
        values="Estimate (95% CI)",
        aggfunc="first",
    )
    wide = wide.reindex(
        pd.MultiIndex.from_product(
            [[NAME_MAP[m] for m in MODELS], [METRIC_LABELS[k] for k in METRICS]]
        )
    )
    wide = wide[["label", "athlete"]]
    wide.columns = ["Label bootstrap (Table 3)", "Athlete bootstrap"]
    return comparison, wide


def log_ci_summary(comparison: pd.DataFrame) -> None:
    """Log how the interval width and the above-baseline conclusion change with athlete resampling."""
    widths = comparison.pivot_table(
        index=["Model", "Metric"], columns="Bootstrap unit", values="CI width"
    )
    widths["ratio athlete/label"] = widths["athlete"] / widths["label"]
    logger.info("CI width ratio (athlete / label):\n%s", widths.round(2).to_string())

    changed = comparison.pivot_table(
        index=["Model", "Metric"],
        columns="Bootstrap unit",
        values="Above baseline",
        aggfunc="first",
    )
    changed["conclusion changed"] = changed["athlete"] != changed["label"]
    logger.info(
        "Does the above-baseline conclusion change with athlete-level resampling?\n%s",
        changed.to_string(),
    )


def domain_shift_ci_table(
    data: Dict[str, CohortData],
    paths: Dict[str, ResultPaths],
    boot_results: Dict[str, np.ndarray],
    point: Dict[str, np.ndarray],
    valid_masks: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """Athlete-level intervals for the domain shift (PhysioNet/CinC 2021 minus LMU).

    The PhysioNet macro value is held fixed while the LMU cohort is resampled.
    """
    rows = []
    for m in MODELS:
        rep = load_per_label_report(paths[m].physionet_report)
        lmu_labels = list(data[m].y_true.columns)
        for k, metric in enumerate(METRICS):
            phys_vals = pd.to_numeric(rep[metric], errors="coerce").reindex(lmu_labels).values
            pair_mask = valid_masks[m][:, k] & ~np.isnan(phys_vals)
            # the LMU-valid label set used in the bootstrap must equal the paired set (as in Table 7)
            same_set = pair_mask.sum() == valid_masks[m][:, k].sum()
            phys_macro = phys_vals[pair_mask].mean()
            lo, hi = percentile_ci(phys_macro - boot_results[m][:, k])
            rows.append(
                {
                    "Model": NAME_MAP[m],
                    "Metric": METRIC_LABELS[metric],
                    "n_pairs": int(pair_mask.sum()),
                    "PhysioNet macro": phys_macro,
                    "LMU macro": point[m][k],
                    "Mean diff": phys_macro - point[m][k],
                    "CI_lo": lo,
                    "CI_hi": hi,
                    "Excludes 0": bool(lo > 0),
                    "same label set as Table 3": same_set,
                }
            )
    shift_ci = pd.DataFrame(rows)
    if not shift_ci["same label set as Table 3"].all():
        logger.warning(
            "Pair set differs from the LMU-valid set for some metrics; re-run the bootstrap on the pair set."
        )
    shift_ci["Mean diff (95% CI)"] = shift_ci.apply(
        lambda r: f"{r['Mean diff']:.3f} ({r['CI_lo']:.3f}–{r['CI_hi']:.3f})", axis=1
    )
    return shift_ci


def model_difference_ci_table(
    boot_results: Dict[str, np.ndarray], point: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """Paired athlete-level intervals for the between-model differences.

    All models are resampled with the same seed and cohort size, so the draws are paired and
    ``boot_results[m1] - boot_results[m2]`` is a valid paired sample.
    """
    rows = []
    for m1, m2 in MODEL_PAIRS:
        for k, metric in enumerate(METRICS):
            lo, hi = percentile_ci(boot_results[m1][:, k] - boot_results[m2][:, k])
            rows.append(
                {
                    "Comparison": f"{NAME_MAP[m1]} − {NAME_MAP[m2]}",
                    "Metric": METRIC_LABELS[metric],
                    "Diff": point[m1][k] - point[m2][k],
                    "CI_lo": lo,
                    "CI_hi": hi,
                    "Excludes 0": bool(lo > 0 or hi < 0),
                }
            )
    model_ci = pd.DataFrame(rows)
    model_ci["Diff (95% CI)"] = model_ci.apply(
        lambda r: f"{r['Diff']:.3f} ({r['CI_lo']:.3f}–{r['CI_hi']:.3f})", axis=1
    )
    return model_ci


def positive_case_probability_table(data: Dict[str, CohortData]) -> pd.DataFrame:
    """Predicted probabilities of the positive cases per label.

    Shows whether undetected findings are confidently negative (near 0) or borderline (near 0.5).
    """
    rows = []
    for m in MODELS:
        y_true, y_prob = data[m].y_true, data[m].y_prob
        for label in POSITIVE_CASE_LABELS:
            if label not in y_true.columns:
                continue
            p = y_prob.loc[y_true[label] == 1, label].values
            rows.append(
                {
                    "Model": NAME_MAP[m],
                    "Label": label,
                    "n_pos": len(p),
                    "Median": np.median(p),
                    "Q1": np.percentile(p, 25),
                    "Q3": np.percentile(p, 75),
                    "% < 0.1": 100 * np.mean(p < 0.1),
                    "% in [0.3, 0.7]": 100 * np.mean((p >= 0.3) & (p <= 0.7)),
                    "% >= 0.5": 100 * np.mean(p >= 0.5),
                }
            )
    pos_prob = pd.DataFrame(rows)
    pos_prob["Median (IQR)"] = pos_prob.apply(
        lambda r: f"{r['Median']:.2f} ({r['Q1']:.2f}–{r['Q3']:.2f})", axis=1
    )
    return pos_prob


def nsr_cooccurrence_table(data: Dict[str, CohortData]) -> pd.DataFrame:
    """NSR probability by label co-occurrence.

    Tests whether the models learned the PhysioNet semantics (NSR as an exclusive normal rhythm):
    the NSR probability should then be high in athletes without any other label and low in athletes
    labelled SB or Brady.
    """
    y_true_ref = data["founder"].y_true  # the ground truth is identical across models
    other_labels = [c for c in y_true_ref.columns if c != "NSR"]
    n_other = y_true_ref[other_labels].sum(axis=1)
    slow_rhythm = (y_true_ref.get("SB", 0) == 1) | (y_true_ref.get("Brady", 0) == 1)
    groups = {
        "NSR only (no other label)": n_other == 0,
        "NSR + SB or Brady": slow_rhythm,
        "NSR + other labels, no SB/Brady": (n_other > 0) & ~slow_rhythm,
    }
    rows = []
    for group, mask in groups.items():
        for m in MODELS:
            p = data[m].y_prob.loc[mask, "NSR"].values
            rows.append(
                {
                    "Group": group,
                    "n": int(mask.sum()),
                    "Model": NAME_MAP[m],
                    "Median (IQR)": f"{np.median(p):.2f} ({np.percentile(p, 25):.2f}–{np.percentile(p, 75):.2f})",
                    "% >= 0.5": round(100 * np.mean(p >= 0.5)),
                }
            )
    return pd.DataFrame(rows)


def log_labels_predicted_instead_of_nsr(data: Dict[str, CohortData]) -> None:
    """Log which labels are predicted positive in athletes whose NSR probability is below 0.5."""
    for m in MODELS:
        y_prob = data[m].y_prob
        missed = y_prob["NSR"] < 0.5
        predicted = (y_prob.loc[missed] >= 0.5).mean().sort_values(ascending=False)
        logger.info(
            "%s: labels predicted positive in the %d athletes with NSR < 0.5\n%s",
            NAME_MAP[m],
            missed.sum(),
            predicted[predicted > 0].round(2).head(6).to_string(),
        )


# --------------------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------------------
def apply_plot_style() -> None:
    sns.set_style("white")
    mpl.rcParams["xtick.bottom"] = True
    mpl.rcParams["ytick.left"] = True
    plt.rc("font", family=STYLE_CONFIG["font_family"])
    plt.rc("axes", linewidth=STYLE_CONFIG["bar_linewidth"])


def _zero_formatter(x: float, pos: int) -> str:
    return f"{x:.1f}" if x != 0 else "0"


def plot_ci_comparison(
    comparison: pd.DataFrame, baselines: Dict[str, Dict[str, float]], out_path: Path
) -> None:
    """Plot label-level against athlete-level confidence intervals for every metric and model."""
    apply_plot_style()
    fig, axes = plt.subplots(
        nrows=1, ncols=len(METRICS), figsize=(3.2 * len(METRICS), 4.2), sharey=True
    )
    offset = 0.18
    model_names = [NAME_MAP[m] for m in MODELS]

    for k, metric in enumerate(METRICS):
        ax = axes[k]
        sub = comparison[comparison["Metric"] == METRIC_LABELS[metric]]
        for i, m in enumerate(MODELS):
            for unit, dx, marker, face in [
                ("label", -offset, "o", "white"),
                ("athlete", +offset, "s", STYLE_CONFIG["model_colors"][i]),
            ]:
                r = sub[(sub["Model"] == NAME_MAP[m]) & (sub["Bootstrap unit"] == unit)].iloc[0]
                ax.errorbar(
                    i + dx,
                    r["Estimate"],
                    yerr=[[r["Estimate"] - r["CI_lo"]], [r["CI_hi"] - r["Estimate"]]],
                    fmt=marker,
                    color="black",
                    markerfacecolor=face,
                    markersize=7,
                    elinewidth=STYLE_CONFIG["bar_linewidth"],
                    capsize=3,
                    capthick=STYLE_CONFIG["bar_linewidth"],
                    zorder=3,
                )
        ax.axhline(
            baselines["ecgfm"][metric],
            color=STYLE_CONFIG["line_color"],
            linestyle=STYLE_CONFIG["line_style"],
            linewidth=0.8,
        )
        ax.set_title(
            METRIC_LABELS[metric], fontsize=STYLE_CONFIG["title_fontsize"], fontweight="bold"
        )
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels(
            model_names, rotation=30, ha="right", fontsize=STYLE_CONFIG["tick_fontsize"]
        )
        ax.set_ylim(0, 1.02)
        ax.yaxis.set_major_formatter(FuncFormatter(_zero_formatter))
        ax.grid(
            axis="y", linestyle=STYLE_CONFIG["grid_linestyle"], alpha=STYLE_CONFIG["grid_alpha"]
        )
        ax.grid(axis="x", visible=False)
        ax.tick_params(
            axis="both",
            direction=STYLE_CONFIG["tick_direction"],
            width=STYLE_CONFIG["tick_width"],
            length=STYLE_CONFIG["tick_length"],
            labelsize=STYLE_CONFIG["tick_fontsize"],
        )
        for spine in ax.spines.values():
            spine.set_edgecolor("#000000")

    axes[0].set_ylabel(
        "Macro metric (95% CI)", fontsize=STYLE_CONFIG["label_fontsize"], fontweight="bold"
    )
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            markerfacecolor="white",
            linestyle="None",
            label="Label bootstrap (Table 3)",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="black",
            markerfacecolor="grey",
            linestyle="None",
            label="Athlete bootstrap",
        ),
        Line2D([0], [0], color="black", linestyle="--", label="Baseline"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=True,
        fontsize=STYLE_CONFIG["tick_fontsize"],
        bbox_to_anchor=(0.5, -0.06),
    )
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    logger.info("Saved figure: %s", out_path)


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
        "--labels",
        type=Path,
        default=DEFAULT_LABELS_CSV,
        help="athlete-level LMU labels",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=DEFAULT_TABLES_DIR,
        help="output directory for CSV tables (default: reports/tables)",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help="output directory for figures (default: reports/figures)",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=N_BOOT,
        help=f"number of bootstrap iterations (default: {N_BOOT}, as in the paper)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    paths = result_paths(args.results_dir)
    check_inputs(args.labels, paths)
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Data and point estimates
    data = load_cohort(load_true_labels(args.labels), paths)
    point, valid_masks = point_estimates(data)
    baselines = metric_baselines(data, valid_masks)

    estimates = pd.DataFrame(
        {NAME_MAP[m]: point[m] for m in MODELS}, index=[METRIC_LABELS[k] for k in METRICS]
    ).T
    n_valid = pd.DataFrame(
        {NAME_MAP[m]: valid_masks[m].sum(axis=0) for m in MODELS}, index=METRICS
    ).T
    logger.info("Point estimates (should reproduce Table 3):\n%s", estimates.round(2).to_string())
    logger.info(
        "Number of valid labels per metric (should reproduce n in Table 3):\n%s",
        n_valid.to_string(),
    )

    # 2. Athlete-level bootstrap (same seed for every model, so the draws are paired)
    boot_results: Dict[str, np.ndarray] = {}
    for m in MODELS:
        logger.info("Bootstrapping %s ...", NAME_MAP[m])
        boot_results[m] = bootstrap_athletes(data[m], valid_masks[m], n_boot=args.n_boot)
    athlete_ci = athlete_ci_table(boot_results, point, valid_masks, baselines)
    logger.info("Athlete-level CIs:\n%s", athlete_ci.round(3).to_string())

    # 3. Comparison with the label-level bootstrap of Table 3
    label_ci = label_ci_table(paths, baselines, n_boot=args.n_boot)
    comparison, wide = compare_bootstrap_units(label_ci, athlete_ci)
    logger.info("Label vs. athlete bootstrap:\n%s", wide.to_markdown())
    comparison.to_csv(args.tables_dir / "bootstrap_athlete_vs_label.csv", index=False)
    wide.to_csv(args.tables_dir / "bootstrap_athlete_vs_label_table.csv")
    plot_ci_comparison(
        comparison, baselines, args.figures_dir / "bootstrap_athlete_vs_label_ci.png"
    )
    log_ci_summary(comparison)

    # 4. Domain shift and between-model differences
    shift_ci = domain_shift_ci_table(data, paths, boot_results, point, valid_masks)
    logger.info(
        "Domain shift (PhysioNet − LMU), athlete-level bootstrap CIs:\n%s",
        shift_ci.pivot(index="Metric", columns="Model", values="Mean diff (95% CI)")
        .reindex([METRIC_LABELS[k] for k in METRICS])
        .to_markdown(),
    )
    shift_ci.to_csv(args.tables_dir / "bootstrap_athlete_domain_shift_ci.csv", index=False)

    model_ci = model_difference_ci_table(boot_results, point)
    logger.info(
        "Between-model differences on LMU, paired athlete-level bootstrap CIs:\n%s",
        model_ci.pivot(index="Metric", columns="Comparison", values="Diff (95% CI)")
        .reindex([METRIC_LABELS[k] for k in METRICS])
        .to_markdown(),
    )
    model_ci.to_csv(args.tables_dir / "bootstrap_athlete_model_diff_ci.csv", index=False)

    # 5. Predicted probabilities of positive cases
    pos_prob = positive_case_probability_table(data)
    pos_wide = pos_prob.pivot(index="Label", columns="Model", values="Median (IQR)").reindex(
        POSITIVE_CASE_LABELS
    )
    pos_wide.insert(
        0, "n_pos", pos_prob.groupby("Label")["n_pos"].first().reindex(POSITIVE_CASE_LABELS)
    )
    logger.info("Predicted probabilities for positive cases (LMU):\n%s", pos_wide.to_markdown())
    pos_prob.to_csv(args.tables_dir / "positive_case_probabilities.csv", index=False)

    # 6. NSR probability by label co-occurrence
    nsr = nsr_cooccurrence_table(data)
    logger.info(
        "NSR probability by label co-occurrence:\n%s",
        nsr.pivot(index=["Group", "n"], columns="Model", values="Median (IQR)").to_markdown(),
    )
    nsr.to_csv(args.tables_dir / "nsr_by_cooccurrence.csv", index=False)
    log_labels_predicted_instead_of_nsr(data)

    logger.info("Tables written to %s", args.tables_dir)


if __name__ == "__main__":
    main()