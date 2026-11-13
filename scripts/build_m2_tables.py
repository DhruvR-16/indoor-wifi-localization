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
    index = {
        (r["exp_id"], r["fold"], r["seed"]): Path(r["predictions_file"])
        for r in records
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
                "delta_mean_position_error": float(
                    per_scan["delta_position_error"].mean()
                ),
                "improves_reference": bool(high < 0.0),
                "degrades_reference": bool(low > 0.0),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("delta_evaal_mean").reset_index(drop=True)


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--reference", default="a4_linear")
    parser.add_argument("--bootstrap", type=int, default=1000)
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
            print(
                f"  {row['exp_id']:<22} dEvAAL {row['delta_evaal_mean']:+6.2f} m "
                f"[{row['delta_evaal_ci_low']:+6.2f}, {row['delta_evaal_ci_high']:+6.2f}] "
                f"| per-run {row['delta_evaal_run_mean']:+6.2f} +/- "
                f"{row['delta_evaal_run_std']:4.2f} m "
                f"| {row['clusters_userid']} users -> {verdict}"
            )
    print(f"\nWrote tables to {results_dir}")


if __name__ == "__main__":
    main()
