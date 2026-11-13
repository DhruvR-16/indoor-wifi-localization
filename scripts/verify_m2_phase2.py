"""Audit Member 2's Phase 2 artifacts for the invariants the protocol depends on.

Checks the data path and every run record/prediction file on disk, independently
of the unit tests, and exits non-zero if any invariant fails. Safe to re-run.

python scripts/verify_m2_phase2.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.load import MISSING_RSSI, load_data  # noqa: E402
from src.data.phase2_m2 import load_m2_fold_data, make_inner_split  # noqa: E402
import src.features.rssi_representations as reps  # noqa: E402

RESULTS_DIR = ROOT / "results" / "phase2" / "m2"
SPLITS_PATH = ROOT / "data" / "splits" / "folds.json"

CLEANED_TRAINING_ROWS = 19_300


class Audit:
    def __init__(self) -> None:
        self.passed: list[str] = []
        self.failed: list[str] = []

    def check(self, condition: bool, label: str, detail: str = "") -> None:
        if condition:
            self.passed.append(label)
            print(f"  PASS  {label}")
        else:
            self.failed.append(f"{label}: {detail}")
            print(f"  FAIL  {label}  {detail}")


def verify_representations(audit: Audit, train_df: pd.DataFrame) -> None:
    print("\n[1] RSSI representations on real scans")
    sample = train_df.filter(like="WAP").head(500).to_numpy(dtype=np.float32)
    observed = sample != float(MISSING_RSSI)
    for name in reps.REPRESENTATIONS:
        values = reps.RepresentationSpec(name)(sample)
        audit.check(
            bool(np.all(values[~observed] == 0.0)),
            f"{name}: not-detected entries map to exactly 0.0",
        )
        audit.check(
            bool(np.all(values[observed] > 0.0)),
            f"{name}: detected readings stay strictly positive",
        )
        audit.check(
            bool(values.min() >= 0.0 and values.max() <= 1.0),
            f"{name}: values bounded to [0, 1]",
            f"range [{values.min():.4f}, {values.max():.4f}]",
        )


def verify_fold_isolation(audit: Audit, train_df: pd.DataFrame) -> None:
    print("\n[2] Fold isolation, coordinate scaler and inner-split coverage")
    folds = json.loads(SPLITS_PATH.read_text())
    spec = reps.RepresentationSpec("linear")

    for fold_info in folds["folds"]:
        fold = fold_info["fold"]
        train_ds, valid_ds, scaler, _ = load_m2_fold_data(
            fold_id=fold, spec=spec, seed=42
        )

        train_users = set(int(u) for u in train_ds.user_ids)
        valid_users = set(int(u) for u in valid_ds.user_ids)
        audit.check(
            not (train_users & valid_users),
            f"fold {fold}: no USERID appears in both partitions",
            f"overlap {sorted(train_users & valid_users)}",
        )
        audit.check(
            valid_ds.augmenter is None,
            f"fold {fold}: held-out partition carries no augmenter",
        )

        expected_mean = (
            train_df.iloc[fold_info["train_idx"]][["LONGITUDE", "LATITUDE"]]
            .to_numpy()
            .mean(axis=0)
        )
        audit.check(
            bool(np.allclose(scaler.mean_, expected_mean)),
            f"fold {fold}: CoordinateScaler fitted on training users only",
            f"{scaler.mean_} vs {expected_mean}",
        )

        outer_train = train_df.iloc[fold_info["train_idx"]]
        _, _, report = make_inner_split(
            user_ids=outer_train["USERID"].to_numpy(),
            buildings=outer_train["BUILDINGID"].to_numpy(),
            floors=outer_train["FLOOR"].to_numpy(),
            mode="stratified",
            seed=42,
        )
        audit.check(
            report.is_covered and not report.floors_missing_from_inner_train,
            f"fold {fold}: inner training covers every inner-validation label",
            f"buildings {report.buildings_missing_from_inner_train} "
            f"floors {report.floors_missing_from_inner_train}",
        )
        audit.check(
            not set(report.inner_train_users) & set(report.inner_val_users),
            f"fold {fold}: inner split keeps users intact",
        )


def verify_records(audit: Audit, records_dir: Path) -> list[dict]:
    print("\n[3] Run records")
    records = []
    for path in sorted(records_dir.glob("record_*.json")):
        records.append(json.loads(path.read_text()))
    audit.check(bool(records), "run records exist", str(records_dir))
    if not records:
        return records

    folds = json.loads(SPLITS_PATH.read_text())
    valid_rows = {f["fold"]: len(f["valid_idx"]) for f in folds["folds"]}

    bad_rows, bad_split, bad_oracle, bad_stub = [], [], [], []
    for record in records:
        name = f"{record['exp_id']}_f{record['fold']}_s{record['seed']}"
        if record["metrics"]["n_samples"] != valid_rows[record["fold"]]:
            bad_rows.append(name)
        inner = record.get("inner_split", {})
        if inner.get("mode") == "stratified" and inner.get(
            "buildings_missing_from_inner_train"
        ):
            bad_split.append(name)
        estimator = record.get("estimator") or {}
        if estimator.get("routed_by") == "oracle" and not estimator.get(
            "diagnostic_only"
        ):
            bad_oracle.append(name)
        if record["exp_id"].startswith("a8_") and not record.get("blocked_on_m3"):
            bad_stub.append(name)

    audit.check(
        not bad_rows,
        "every run scored exactly its fold's held-out rows",
        str(bad_rows[:3]),
    )
    audit.check(
        not bad_split,
        "no stratified run used an uncovered inner split",
        str(bad_split[:3]),
    )
    audit.check(
        not bad_oracle,
        "oracle-routed runs are flagged diagnostic_only",
        str(bad_oracle[:3]),
    )
    audit.check(not bad_stub, "A8 records declare blocked_on_m3", str(bad_stub[:3]))
    return records


def verify_predictions(
    audit: Audit, records: list[dict], train_df: pd.DataFrame
) -> None:
    print("\n[4] Prediction files")
    if not records:
        return
    folds = json.loads(SPLITS_PATH.read_text())
    fold_users = {
        f["fold"]: set(int(u) for u in train_df.iloc[f["valid_idx"]]["USERID"].unique())
        for f in folds["folds"]
    }

    missing, wrong_users, bad_error = [], [], []
    for record in records:
        path = Path(record["predictions_file"])
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            missing.append(path.name)
            continue
        frame = pd.read_csv(path)
        if set(frame["USERID"].unique()) != fold_users[record["fold"]]:
            wrong_users.append(path.name)
        recomputed = np.hypot(
            frame["LONGITUDE_PRED"] - frame["LONGITUDE_TRUE"],
            frame["LATITUDE_PRED"] - frame["LATITUDE_TRUE"],
        )
        if not np.allclose(recomputed, frame["position_error"], atol=1e-6):
            bad_error.append(path.name)

    audit.check(not missing, "every record's prediction file exists", str(missing[:3]))
    audit.check(
        not wrong_users,
        "prediction rows belong only to that fold's held-out users",
        str(wrong_users[:3]),
    )
    audit.check(
        not bad_error,
        "stored position_error matches the recomputed distance",
        str(bad_error[:3]),
    )


def verify_pooled_coverage(audit: Audit, records: list[dict]) -> None:
    print("\n[5] Pooled out-of-fold coverage")
    by_arm_seed: dict[tuple[str, int], list[dict]] = {}
    for record in records:
        by_arm_seed.setdefault((record["exp_id"], record["seed"]), []).append(record)

    complete = {k: v for k, v in by_arm_seed.items() if len(v) == 5}
    if not complete:
        print("  SKIP  no arm has all five folds yet")
        return
    bad = []
    for (exp_id, seed), group in complete.items():
        total = sum(r["metrics"]["n_samples"] for r in group)
        if total != CLEANED_TRAINING_ROWS:
            bad.append(f"{exp_id} s{seed}={total}")
    audit.check(
        not bad,
        f"each complete arm/seed covers all {CLEANED_TRAINING_ROWS:,} cleaned rows once",
        str(bad[:3]),
    )


def verify_dae_checkpoints(audit: Audit, dae_dir: Path) -> None:
    print("\n[6] DAE checkpoints")
    summaries = sorted(dae_dir.glob("dae_*.json"))
    if not summaries:
        print("  SKIP  no DAE checkpoints yet")
        return
    bad_fold, bad_dim, bad_recon = [], [], []
    for path in summaries:
        summary = json.loads(path.read_text())
        stem_fold = int(path.stem.split("_f")[1].split("_s")[0])
        stem_seed = int(path.stem.split("_s")[1])
        if summary["fold"] != stem_fold or summary["seed"] != stem_seed:
            bad_fold.append(path.name)
        if summary["input_dim"] != 520:
            bad_dim.append(path.name)
        recon = summary["reconstruction"]
        if not (
            recon["inner_val_observed_mse"] > 0 and recon["inner_val_missing_mse"] >= 0
        ):
            bad_recon.append(path.name)
    audit.check(
        not bad_fold, "checkpoint fold/seed match their filenames", str(bad_fold[:3])
    )
    audit.check(not bad_dim, "checkpoints are 520-wide", str(bad_dim[:3]))
    audit.check(
        not bad_recon,
        "observed and missing reconstruction are both recorded",
        str(bad_recon[:3]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    print("Verifying Member 2 Phase 2 artifacts")
    train_df, _, _ = load_data()
    audit = Audit()

    verify_representations(audit, train_df)
    verify_fold_isolation(audit, train_df)
    records = verify_records(audit, results_dir / "records")
    verify_predictions(audit, records, train_df)
    verify_pooled_coverage(audit, records)
    verify_dae_checkpoints(audit, results_dir / "dae")

    print(f"\n{len(audit.passed)} checks passed, {len(audit.failed)} failed")
    for failure in audit.failed:
        print(f"  - {failure}")
    return 1 if audit.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
