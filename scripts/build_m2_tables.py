"""Aggregate Member 2's Phase 2 run records into comparison tables.

Produces, under results/phase2/m2/:

* ``m2_all_runs.csv``        - one row per (arm, fold, seed) with every metric.
* ``m2_summary_metrics.csv`` - per arm, mean +/- std across folds and seeds,
  plus the pooled out-of-fold estimate per seed averaged over seeds.
* ``m2_paired_comparison.csv`` - each arm against the reference arm on the
  primary metric, paired on identical held-out rows, with a USERID-clustered
  bootstrap confidence interval.

Pairing is row-level: arms are only compared where they predicted the same
(fold, row index) under the same seed, which the A4/A7 design guarantees.

Example
-------
python scripts/build_m2_tables.py --reference a4_linear
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import evaluate_predictions  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "phase2_m2.yaml"
RESULTS_DIR = ROOT / "results" / "phase2" / "m2"

HEADLINE_METRICS = [
    "building_accuracy",
    "building_macro_f1",
    "floor_accuracy",
    "floor_macro_f1",
    "floor_accuracy_given_correct_building",
    "mae_longitude",
    "mae_latitude",
    "mean_position_error",
    "median_position_error",
    "p95_position_error",
    "evaal_style_score",
]


def load_records(records_dir: Path) -> list[dict]:
    records = []
    for path in sorted(records_dir.glob("record_*.json")):
        with open(path) as handle:
            records.append(json.load(handle))
    return records


def runs_table(records: list[dict]) -> pd.DataFrame:
    rows = []
    for record in records:
        row = {
            "exp_id": record["exp_id"],
            "fold": record["fold"],
            "seed": record["seed"],
            "representation": record.get("representation"),
            "include_missing_mask": record.get("include_missing_mask"),
            "input_dim": record.get("input_dim"),
            "pretrained": record.get("pretraining", {}).get("pretrained", False),
            "inner_split_mode": record.get("inner_split", {}).get("mode"),
            "inner_split_covered": not record.get("inner_split", {}).get(
                "buildings_missing_from_inner_train", []
            ),
            "best_epoch": record["best_epoch"],
            "best_inner_evaal": record["best_inner_evaal"],
            "final_train_loss": record["final_train_loss"],
            "final_val_loss": record["final_val_loss"],
            "ablation": record.get("config", {}).get("notes", {}).get("ablation"),
        }
        row.update(record["metrics"])
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values(["exp_id", "fold", "seed"])
        .reset_index(drop=True)
    )


def pooled_oof_by_seed(records: list[dict]) -> pd.DataFrame:
    """Pool the five folds' predictions per (arm, seed) and score once.

    This is the Phase 1 convention: every cleaned training row appears exactly
    once per seed, so the pooled estimate is not a mean of fold means.
    """
    rows = []
    frame = pd.DataFrame(
        [
            {
                "exp_id": r["exp_id"],
                "seed": r["seed"],
                "fold": r["fold"],
                "preds": r["predictions_file"],
            }
            for r in records
        ]
    )
    for (exp_id, seed), group in frame.groupby(["exp_id", "seed"]):
        parts = []
        for _, item in group.iterrows():
            path = Path(item["preds"])
            if not path.is_absolute():
                path = ROOT / path
            if not path.exists():
                continue
            part = pd.read_csv(path)
            part["fold"] = item["fold"]
            parts.append(part)
        if not parts:
            continue
        pooled = pd.concat(parts, ignore_index=True)
        metrics = evaluate_predictions(pooled)
        rows.append(
            {
                "exp_id": exp_id,
                "seed": seed,
                "folds_pooled": len(parts),
                "pooled_rows": len(pooled),
                **{k: metrics[k] for k in HEADLINE_METRICS if k in metrics},
            }
        )
    return pd.DataFrame(rows).sort_values(["exp_id", "seed"]).reset_index(drop=True)


def summary_table(runs: pd.DataFrame, pooled: pd.DataFrame) -> pd.DataFrame:
    present = [m for m in HEADLINE_METRICS if m in runs.columns]
    grouped = runs.groupby("exp_id")
    summary = pd.DataFrame(
        {
            "exp_id": [name for name, _ in grouped],
            "runs": [len(group) for _, group in grouped],
            "folds": [group["fold"].nunique() for _, group in grouped],
            "seeds": [group["seed"].nunique() for _, group in grouped],
            "mean_best_epoch": [group["best_epoch"].mean() for _, group in grouped],
        }
    )
    for metric in present:
        summary[f"{metric}_mean"] = [group[metric].mean() for _, group in grouped]
        summary[f"{metric}_std"] = [group[metric].std(ddof=1) for _, group in grouped]

    if not pooled.empty:
        pooled_grouped = pooled.groupby("exp_id")
        pooled_stats = pd.DataFrame({"exp_id": [name for name, _ in pooled_grouped]})
        for metric in [m for m in HEADLINE_METRICS if m in pooled.columns]:
            pooled_stats[f"pooled_{metric}_mean"] = [
                group[metric].mean() for _, group in pooled_grouped
            ]
            pooled_stats[f"pooled_{metric}_std"] = [
                group[metric].std(ddof=1) for _, group in pooled_grouped
            ]
        summary = summary.merge(pooled_stats, on="exp_id", how="left")
    return summary.sort_values("exp_id").reset_index(drop=True)


def _row_level_primary(path: Path) -> pd.DataFrame:
    """Per-row EvAAL contribution, which is what the primary metric averages."""
    frame = pd.read_csv(path if path.is_absolute() else ROOT / path)
    building_wrong = frame["BUILDINGID_PRED"] != frame["BUILDINGID_TRUE"]
    floor_wrong = frame["FLOOR_PRED"] != frame["FLOOR_TRUE"]
    frame["evaal_row"] = (
        frame["position_error"] + 50.0 * building_wrong + 4.0 * floor_wrong
    )
    return frame[["index", "USERID", "evaal_row", "position_error"]]


def paired_comparison(
    records: list[dict],
    reference: str,
    n_boot: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Paired difference vs the reference arm, with a user-clustered bootstrap.

    Two levels of correlation make a row-level iid bootstrap wrong here: rows
    predicted by one model are not independent of each other, and the same
    physical scan reappears under every seed. So the per-row difference is
    first averaged over seeds, leaving one value per scan, and the bootstrap
    then resamples whole held-out USERIDs with replacement - the users are the
    independent replicates of this design. The run-level mean and standard
    deviation over (fold, seed) runs are reported alongside, since that is the
    spread a reader of the summary table sees.
    """
    # Oracle-routed arms are diagnostics, not end-to-end results: they stay in
    # the summary table but never in the headline comparison.
    diagnostic_arms = {
        r["exp_id"]
        for r in records
        if (r.get("estimator") or {}).get("diagnostic_only")
    }
    index = {
        (r["exp_id"], r["fold"], r["seed"]): Path(r["predictions_file"])
        for r in records
        if r["exp_id"] not in diagnostic_arms
    }
    reference_keys = {k for k in index if k[0] == reference}
    if not reference_keys:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    rows = []
    arms = sorted({k[0] for k in index} - {reference})
    for arm in arms:
        per_run, frames = [], []
        for _, fold, run_seed in sorted(reference_keys, key=lambda k: (k[1], k[2])):
            arm_key = (arm, fold, run_seed)
            if arm_key not in index:
                continue
            ref_rows = _row_level_primary(index[(reference, fold, run_seed)])
            arm_rows = _row_level_primary(index[arm_key])
            merged = ref_rows.merge(
                arm_rows, on=["index", "USERID"], suffixes=("_ref", "_arm")
            )
            if len(merged) != len(ref_rows):
                raise AssertionError(
                    f"{arm} and {reference} do not cover identical rows on fold {fold} "
                    f"seed {run_seed}; refusing to compare unmatched rows."
                )
            merged["delta_evaal"] = merged["evaal_row_arm"] - merged["evaal_row_ref"]
            merged["delta_position_error"] = (
                merged["position_error_arm"] - merged["position_error_ref"]
            )
            merged["fold"] = fold
            frames.append(
                merged[
                    ["fold", "index", "USERID", "delta_evaal", "delta_position_error"]
                ]
            )
            per_run.append(float(merged["delta_evaal"].mean()))
        if not frames:
            continue

        pooled = pd.concat(frames, ignore_index=True)
        # One value per physical scan: average the seeds, which are repeats of it.
        per_scan = pooled.groupby(["fold", "index"], as_index=False).agg(
            USERID=("USERID", "first"),
            delta_evaal=("delta_evaal", "mean"),
            delta_position_error=("delta_position_error", "mean"),
        )
        by_user = {
            user: group["delta_evaal"].to_numpy()
            for user, group in per_scan.groupby("USERID")
        }
        users = np.array(sorted(by_user))
        boot = np.empty(n_boot)
        for draw in range(n_boot):
            picked = rng.choice(users, size=len(users), replace=True)
            boot[draw] = np.concatenate([by_user[u] for u in picked]).mean()
        low, high = np.percentile(boot, [2.5, 97.5])

        # Users are nested in folds and share a fold's model pair, so the
        # user-clustered interval is still optimistic for a model-level claim.
        # The five outer folds are the independent replicates, so report their
        # spread and win rate too, and require both to agree before calling a
        # difference robust.
        # Row-weighted within each fold, because EvAAL is itself a row mean -
        # weighting users equally here would report a different estimand from
        # the metric in the summary table.
        fold_values = per_scan.groupby("fold")["delta_evaal"].mean().to_numpy()
        # With only five folds a percentile bootstrap is unreliable - it called
        # a4_powed_mask and a7_dae_pretrained equally "robust" when a paired
        # test over the fold means puts them at p = 0.046 and p = 0.107. Use a
        # t interval over the fold means instead, and report the p-value with
        # it so the verdict and the test cannot disagree.
        n_folds = len(fold_values)
        fold_mean = float(fold_values.mean())
        if n_folds > 1:
            fold_sd = float(np.std(fold_values, ddof=1))
            stderr = fold_sd / np.sqrt(n_folds)
            if stderr > 0.0:
                half_width = float(stats.t.ppf(0.975, n_folds - 1)) * stderr
                fold_p = float(stats.ttest_1samp(fold_values, 0.0).pvalue)
            else:
                half_width, fold_p = 0.0, float("nan")
            fold_low, fold_high = fold_mean - half_width, fold_mean + half_width
        else:
            fold_sd = float("nan")
            fold_low = fold_high = fold_mean
            fold_p = float("nan")
        user_ci_improves = bool(high < 0.0)
        user_ci_degrades = bool(low > 0.0)

        rows.append(
            {
                "exp_id": arm,
                "reference": reference,
                "paired_runs": len(per_run),
                "paired_scans": len(per_scan),
                "clusters_userid": len(users),
                "delta_evaal_mean": float(per_scan["delta_evaal"].mean()),
                "delta_evaal_ci_low": float(low),
                "delta_evaal_ci_high": float(high),
                "delta_evaal_run_mean": float(np.mean(per_run)),
                "delta_evaal_run_std": (
                    float(np.std(per_run, ddof=1)) if len(per_run) > 1 else float("nan")
                ),
                "delta_evaal_fold_mean": float(fold_values.mean()),
                "delta_evaal_fold_std": fold_sd,
                "delta_evaal_fold_ci_low": float(fold_low),
                "delta_evaal_fold_ci_high": float(fold_high),
                "delta_evaal_fold_p_value": fold_p,
                "folds_won": int((fold_values < 0).sum()),
                "folds_total": int(len(fold_values)),
                "runs_won": int(sum(1 for value in per_run if value < 0)),
                "runs_total": int(len(per_run)),
                "delta_mean_position_error": float(
                    per_scan["delta_position_error"].mean()
                ),
                "improves_reference": user_ci_improves,
                "degrades_reference": user_ci_degrades,
                "complete_grid": bool(len(fold_values) >= 5 and len(per_run) >= 15),
                "robust_across_folds": bool(
                    len(fold_values) >= 3
                    and (
                        (user_ci_improves and fold_high < 0.0)
                        or (user_ci_degrades and fold_low > 0.0)
                    )
                ),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("delta_evaal_mean").reset_index(drop=True)


def markdown_tables(
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    pooled: pd.DataFrame,
    primary: str,
    reference: str,
) -> str:
    """Render the report's results tables straight from the aggregated frames.

    The report cites this file so that no figure in it is transcribed by hand.
    """
    lines: list[str] = []
    lines.append(
        f"<!-- Generated by scripts/build_m2_tables.py --reference {reference} -->"
    )
    lines.append("")
    lines.append("### Per-run results (mean +/- std over folds x seeds)")
    lines.append("")
    lines.append(
        "| Arm | Runs | EvAAL (m) | Mean error (m) | Median (m) | P95 (m) "
        "| Building acc (%) | Floor acc (%) | Mean E* |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, row in summary.sort_values(f"{primary}_mean").iterrows():
        lines.append(
            f"| `{row['exp_id']}` | {int(row['runs'])} "
            f"| {row[f'{primary}_mean']:.2f} +/- {row[f'{primary}_std']:.2f} "
            f"| {row['mean_position_error_mean']:.2f} +/- {row['mean_position_error_std']:.2f} "
            f"| {row['median_position_error_mean']:.2f} "
            f"| {row['p95_position_error_mean']:.2f} "
            f"| {row['building_accuracy_mean'] * 100:.2f} "
            f"| {row['floor_accuracy_mean'] * 100:.2f} "
            f"| {row['mean_best_epoch']:.1f} |"
        )

    if not pooled.empty:
        lines += [
            "",
            "### Pooled out-of-fold estimate (all five folds per seed, averaged over seeds)",
            "",
        ]
        lines.append(
            "| Arm | Seeds | Pooled rows | EvAAL (m) | Mean error (m) "
            "| Building acc (%) | Floor acc (%) | Floor given building (%) |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        grouped = pooled.groupby("exp_id")
        for exp_id, group in grouped:
            lines.append(
                f"| `{exp_id}` | {group['seed'].nunique()} "
                f"| {int(group['pooled_rows'].iloc[0]):,} "
                f"| {group[primary].mean():.2f} +/- {group[primary].std(ddof=1):.2f} "
                f"| {group['mean_position_error'].mean():.2f} "
                f"| {group['building_accuracy'].mean() * 100:.2f} "
                f"| {group['floor_accuracy'].mean() * 100:.2f} "
                f"| {group['floor_accuracy_given_correct_building'].mean() * 100:.2f} |"
            )

    if not comparison.empty:
        lines += [
            "",
            f"### Paired comparison against `{reference}` (USERID-clustered bootstrap)",
            "",
            "Negative is better. The interval resamples whole held-out users, which are "
            "the independent replicates of a user-grouped design.",
            "",
            "| Arm | dEvAAL (m) | 95% CI (users) | Per-fold dEvAAL (m) "
            "| 95% CI (folds) | p (folds) | Folds won | Runs won | Verdict |",
            "|---|---:|---|---:|---|---:|---:|---:|---|",
        ]
        for _, row in comparison.iterrows():
            verdict = (
                "improves"
                if row["improves_reference"]
                else "degrades" if row["degrades_reference"] else "inconclusive"
            )
            if row["improves_reference"] or row["degrades_reference"]:
                verdict += (
                    " (robust)" if row["robust_across_folds"] else " (not fold-robust)"
                )
            if not row["complete_grid"]:
                verdict += " [partial grid]"
            lines.append(
                f"| `{row['exp_id']}` | {row['delta_evaal_mean']:+.2f} "
                f"| [{row['delta_evaal_ci_low']:+.2f}, {row['delta_evaal_ci_high']:+.2f}] "
                f"| {row['delta_evaal_fold_mean']:+.2f} +/- {row['delta_evaal_fold_std']:.2f} "
                f"| [{row['delta_evaal_fold_ci_low']:+.2f}, "
                f"{row['delta_evaal_fold_ci_high']:+.2f}] "
                f"| {row['delta_evaal_fold_p_value']:.3f} "
                f"| {int(row['folds_won'])}/{int(row['folds_total'])} "
                f"| {int(row['runs_won'])}/{int(row['runs_total'])} | **{verdict}** |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--reference", default="a4_linear")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument(
        "--markdown",
        nargs="?",
        const="results/phase2/m2/m2_results_tables.md",
        default=None,
        help="Also write the report's results tables as markdown.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    records = load_records(results_dir / "records")
    if not records:
        print(f"No run records found under {results_dir/'records'}")
        return

    runs = runs_table(records)
    runs.to_csv(results_dir / "m2_all_runs.csv", index=False)

    pooled = pooled_oof_by_seed(records)
    if not pooled.empty:
        pooled.to_csv(results_dir / "m2_pooled_oof_by_seed.csv", index=False)

    summary = summary_table(runs, pooled)
    summary.to_csv(results_dir / "m2_summary_metrics.csv", index=False)

    comparison = paired_comparison(records, args.reference, n_boot=args.bootstrap)
    if not comparison.empty:
        comparison.to_csv(results_dir / "m2_paired_comparison.csv", index=False)

    primary = config["primary_metric"]
    print(f"{len(runs)} runs across {runs['exp_id'].nunique()} arms\n")
    print(f"Per-run mean +/- std, primary metric = {primary} (lower is better):")
    for _, row in summary.iterrows():
        print(
            f"  {row['exp_id']:<22} n={int(row['runs']):>2} "
            f"EvAAL {row[f'{primary}_mean']:6.2f} +/- {row[f'{primary}_std']:5.2f} m | "
            f"mean err {row['mean_position_error_mean']:6.2f} m | "
            f"bldg {row['building_accuracy_mean']*100:5.2f}% | "
            f"floor {row['floor_accuracy_mean']*100:5.2f}% | E* {row['mean_best_epoch']:.1f}"
        )
    if not comparison.empty:
        print(
            f"\nPaired comparison vs {args.reference} (negative = better, "
            f"95% CI from a USERID-clustered bootstrap):"
        )
        for _, row in comparison.iterrows():
            verdict = (
                "improves"
                if row["improves_reference"]
                else "degrades" if row["degrades_reference"] else "inconclusive"
            )
            if row["improves_reference"] or row["degrades_reference"]:
                verdict += (
                    " (robust)" if row["robust_across_folds"] else " (not fold-robust)"
                )
            if not row["complete_grid"]:
                verdict += " [partial grid]"
            print(
                f"  {row['exp_id']:<22} dEvAAL {row['delta_evaal_mean']:+6.2f} m "
                f"[{row['delta_evaal_ci_low']:+6.2f}, {row['delta_evaal_ci_high']:+6.2f}] "
                f"| folds {row['delta_evaal_fold_mean']:+6.2f} +/- "
                f"{row['delta_evaal_fold_std']:4.2f} "
                f"[{row['delta_evaal_fold_ci_low']:+6.2f}, "
                f"{row['delta_evaal_fold_ci_high']:+6.2f}] m "
                f"| p={row['delta_evaal_fold_p_value']:.3f} "
                f"| won {row['folds_won']}/{row['folds_total']} folds, "
                f"{row['runs_won']}/{row['runs_total']} runs -> {verdict}"
            )
    if args.markdown:
        markdown_path = Path(args.markdown)
        if not markdown_path.is_absolute():
            markdown_path = ROOT / markdown_path
        markdown_path.write_text(
            markdown_tables(summary, comparison, pooled, primary, args.reference)
        )
        print(f"Wrote markdown tables to {markdown_path}")

    print(f"\nWrote tables to {results_dir}")


if __name__ == "__main__":
    main()
