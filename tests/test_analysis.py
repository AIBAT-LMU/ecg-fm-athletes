"""Unit tests for the shared analysis building blocks (``ecgath.analysis``).

They run on small synthetic inputs and need neither data nor a GPU. The regression tests pin the
exact output of the resampling procedures for a fixed seed: the numbers in the paper depend on the
order in which random numbers are drawn, so a change there must not go unnoticed.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ecgath import analysis

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------
@pytest.fixture
def labels_csv(tmp_path: Path) -> Path:
    """Athlete labels in the raw LMU format (Ja/Nein, unmerged columns)."""
    table = pd.DataFrame(
        {
            "SB": ["Ja", "Nein", "Ja"],
            "CRBBB": ["Nein", "Ja", "Nein"],
            "RBBB": ["Nein", "Nein", "Ja"],
            "PAC": ["Nein", "Nein", "Nein"],
            "SVPB": ["Nein", "Nein", "Nein"],
        },
        index=pd.Index(["A01", "A02", "A03"], name="id"),
    )
    path = tmp_path / "labels.csv"
    table.to_csv(path)
    return path


def per_label_report(values: dict) -> pd.DataFrame:
    """Per-label report with one metric column, indexed by class like the real reports."""
    report = pd.DataFrame({"Class": list(values), "AUROC": list(values.values())})
    return report.set_index("Class", drop=False)


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def test_load_true_labels_converts_and_merges(labels_csv: Path) -> None:
    labels = analysis.load_true_labels(labels_csv)

    assert list(labels.index) == ["A01", "A02", "A03"]
    assert labels["SB"].tolist() == [1, 0, 1]
    # CRBBB and RBBB are merged with a logical OR and the source columns are dropped
    assert labels["CRBBB|RBBB"].tolist() == [0, 1, 1]
    assert labels["PAC|SVPB"].tolist() == [0, 0, 0]
    assert not {"CRBBB", "RBBB", "PAC", "SVPB"} & set(labels.columns)


def test_load_true_labels_without_merging(labels_csv: Path) -> None:
    labels = analysis.load_true_labels(labels_csv, merge_columns=False)

    assert set(labels.columns) == {"SB", "CRBBB", "RBBB", "PAC", "SVPB"}
    assert labels["RBBB"].tolist() == [0, 0, 1]


def test_load_true_labels_casts_numeric_ids_to_str(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    pd.DataFrame({"SB": ["Ja", "Nein"]}, index=pd.Index([101, 102], name="id")).to_csv(path)

    assert list(analysis.load_true_labels(path).index) == ["101", "102"]


def test_load_per_label_report_drops_global_row(tmp_path: Path) -> None:
    path = tmp_path / "report.csv"
    pd.DataFrame({"Class": ["SB", "AF", "GLOBAL (Macro-Avg)"], "AUROC": [0.9, 0.8, 0.85]}).to_csv(
        path, index=False
    )

    report = analysis.load_per_label_report(path)

    assert list(report.index) == ["SB", "AF"]
    assert "Class" in report.columns  # kept as a column as well


@pytest.mark.parametrize(
    "segment_id, expected", [("A01_0", "A01"), ("A01_12", "A01"), ("A01", "A01"), (17, "17")]
)
def test_athlete_id(segment_id, expected) -> None:
    assert analysis.athlete_id(segment_id) == expected


def test_aggregate_segments_to_athletes_averages_per_athlete() -> None:
    segments = pd.DataFrame(
        {"SB": [0.2, 0.4, 0.9], "AF": [0.0, 1.0, 0.5]},
        index=pd.Index(["A01_0", "A01_1", "A02_0"], name="ecg_segment_id"),
    )

    athletes = analysis.aggregate_segments_to_athletes(segments)

    assert list(athletes.index) == ["A01", "A02"]
    assert athletes.loc["A01", "SB"] == pytest.approx(0.3)
    assert athletes.loc["A01", "AF"] == pytest.approx(0.5)
    assert athletes.loc["A02", "SB"] == pytest.approx(0.9)


# --------------------------------------------------------------------------------------
# Input locations
# --------------------------------------------------------------------------------------
def test_result_paths_follow_the_directory_layout() -> None:
    paths = analysis.ResultPaths.for_model(Path("results"), "founder")

    assert paths.raw_segments == Path("results/founder_lmu/raw_segment_predictions.csv")
    assert paths.agg_report == Path("results/founder_lmu/athlete_aggregated_metrics_report.csv")
    assert paths.physionet_report == Path(
        "results/founder_physionet2021/evaluation_results_test.csv"
    )
    assert set(analysis.result_paths(Path("results"))) == set(analysis.MODELS)


def test_check_inputs_lists_all_missing_files(tmp_path: Path) -> None:
    paths = analysis.result_paths(tmp_path / "results")

    with pytest.raises(FileNotFoundError) as error:
        analysis.check_inputs(tmp_path / "labels.csv", paths)

    message = str(error.value)
    assert "labels.csv" in message
    assert message.count("raw_segment_predictions.csv") == len(analysis.MODELS)


# --------------------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------------------
def test_bootstrap_macro_regression() -> None:
    """Exact output for a fixed seed; changes if the random number usage changes."""
    values = [0.2, 0.4, 0.5, 0.7, 0.9, np.nan]

    mean, lower, upper = analysis.bootstrap_macro(values, n_boot=500, seed=42)

    assert mean == pytest.approx(0.54, abs=1e-12)
    assert lower == pytest.approx(0.3295, abs=1e-12)
    assert upper == pytest.approx(0.7505, abs=1e-12)


def test_bootstrap_macro_is_reproducible_and_seed_dependent() -> None:
    values = np.linspace(0.1, 0.9, 12)

    first = analysis.bootstrap_macro(values, n_boot=200, seed=1)

    assert analysis.bootstrap_macro(values, n_boot=200, seed=1) == first
    assert analysis.bootstrap_macro(values, n_boot=200, seed=2) != first


def test_bootstrap_macro_ignores_nan_and_brackets_the_mean() -> None:
    mean, lower, upper = analysis.bootstrap_macro([0.3, np.nan, 0.5, 0.7, np.nan], n_boot=300)

    assert mean == pytest.approx(0.5)
    assert lower <= mean <= upper


@pytest.mark.parametrize("values", [[], [0.5], [np.nan, np.nan], [0.5, np.nan]])
def test_bootstrap_macro_needs_two_valid_values(values) -> None:
    assert all(np.isnan(v) for v in analysis.bootstrap_macro(values, n_boot=50))


# --------------------------------------------------------------------------------------
# Paired permutation test
# --------------------------------------------------------------------------------------
X = [0.90, 0.80, 0.85, 0.70, 0.95, 0.60]
Y = [0.70, 0.75, 0.60, 0.65, 0.80, 0.62]


def test_paired_permutation_test_regression() -> None:
    """Exact output for a fixed seed; changes if the random number usage changes."""
    greater = analysis.paired_permutation_test(X, Y, n_perm=500, seed=42, alternative="greater")
    two_sided = analysis.paired_permutation_test(
        X, Y, n_perm=500, seed=42, alternative="two-sided"
    )

    assert greater["n_pairs"] == 6
    assert greater["d_obs"] == pytest.approx(0.11333333333333333, abs=1e-12)
    assert greater["p_value"] == pytest.approx(0.017964071856287425, abs=1e-12)
    assert two_sided["p_value"] == pytest.approx(0.02594810379241517, abs=1e-12)


def test_paired_permutation_test_direction() -> None:
    greater = analysis.paired_permutation_test(X, Y, n_perm=500, alternative="greater")
    less = analysis.paired_permutation_test(X, Y, n_perm=500, alternative="less")

    assert greater["p_value"] < 0.05 < less["p_value"]


def test_paired_permutation_test_without_difference() -> None:
    result = analysis.paired_permutation_test(X, X, n_perm=200, alternative="greater")

    assert result["d_obs"] == 0.0
    assert result["p_value"] == 1.0


def test_paired_permutation_test_drops_incomplete_pairs() -> None:
    result = analysis.paired_permutation_test([0.9, np.nan, 0.8], [0.5, 0.4, np.nan], n_perm=50)

    assert result["n_pairs"] == 1
    assert result["d_obs"] == pytest.approx(0.4)


def test_paired_permutation_test_without_valid_pairs() -> None:
    result = analysis.paired_permutation_test([np.nan], [0.5], n_perm=50)

    assert result["n_pairs"] == 0
    assert np.isnan(result["p_value"])


# --------------------------------------------------------------------------------------
# Model comparison
# --------------------------------------------------------------------------------------
LABELS = [f"L{i:02d}" for i in range(12)]


def reports_with_offsets(offsets: dict) -> dict:
    """Reports in which every model has the same per-label profile plus a constant offset."""
    base = np.linspace(0.40, 0.62, len(LABELS))
    return {
        model: per_label_report(dict(zip(LABELS, base + offset)))
        for model, offset in offsets.items()
    }


def test_compare_models_detects_a_consistent_ranking() -> None:
    reports = reports_with_offsets({"ecgfm": 0.10, "founder": 0.20, "hubert": 0.0})

    friedman, posthoc = analysis.compare_models(reports, "AUROC")

    assert friedman["n_labels_used"] == len(LABELS)
    assert friedman["friedman_p"] < analysis.ALPHA
    assert posthoc["pair"].tolist() == ["ecgfm vs founder", "ecgfm vs hubert", "founder vs hubert"]
    assert posthoc["significant_fdr"].all()
    # Benjamini-Hochberg never lowers a p-value
    assert (posthoc["p_adj_fdr"] >= posthoc["p_raw"] - 1e-15).all()


def test_compare_models_skips_posthoc_without_overall_difference() -> None:
    rng = np.random.RandomState(0)
    reports = {m: per_label_report(dict(zip(LABELS, rng.rand(12)))) for m in analysis.MODELS}

    friedman, posthoc = analysis.compare_models(reports, "AUROC")

    assert friedman["friedman_p"] >= analysis.ALPHA
    assert posthoc is None


def test_compare_models_excludes_labels_undefined_for_any_model() -> None:
    reports = reports_with_offsets({"ecgfm": 0.10, "founder": 0.20, "hubert": 0.0})
    reports["hubert"].loc["L03", "AUROC"] = np.nan

    friedman, _ = analysis.compare_models(reports, "AUROC")

    assert friedman["n_labels_used"] == len(LABELS) - 1
    assert "L03" not in friedman["labels_used"]


def test_compare_models_needs_two_labels() -> None:
    reports = {m: per_label_report({"L00": 0.5}) for m in analysis.MODELS}

    friedman, posthoc = analysis.compare_models(reports, "AUROC")

    assert np.isnan(friedman["friedman_p"])
    assert posthoc is None


# --------------------------------------------------------------------------------------
# Published reference values
# --------------------------------------------------------------------------------------
def test_paper_values_file_is_well_formed() -> None:
    checks = json.loads((REPO_ROOT / "tests" / "paper_values.json").read_text(encoding="utf-8"))[
        "checks"
    ]

    assert checks
    for check in checks:
        assert check["file"].endswith(".csv")
        assert check["tolerance"] >= 0
        assert check["rows"]
        for row in check["rows"]:
            assert row["key"] and row["values"]