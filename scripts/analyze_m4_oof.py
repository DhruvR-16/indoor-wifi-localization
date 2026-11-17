"""M4 paired analysis of saved Phase 2 OOF predictions.

All uncertainty intervals resample USERID clusters, not individual scans. The
exact McNemar p-value is included as a conventional scan-level diagnostic and
must be interpreted alongside the user-cluster bootstrap interval.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from src.data.load import load_data
from src.data.phase2_m3 import Phase2Dataset
from src.models.dl.multitask import make_model
M3_ARMS = [
    "a1_shared",
    "a1_independent",
    "a2_inner_selected_weights",
    "a2_uncertainty_weights",
    "a3_hierarchical_floor",
]
METADATA = ["USERID", "PHONEID", "BUILDINGID", "FLOOR", "LONGITUDE_TRUE", "LATITUDE_TRUE"]


def _prepare_m3(path: Path, clean_train: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(path)
    # The M3 prediction writer stores true UTM coordinates as float32. Restore
    # source precision via source_row before recomputing metre-scale metrics.
    source = frame.source_row.to_numpy(dtype=int)
    frame["LONGITUDE_TRUE"] = clean_train.iloc[source].LONGITUDE.to_numpy()
    frame["LATITUDE_TRUE"] = clean_train.iloc[source].LATITUDE.to_numpy()
    frame["position_error"] = np.hypot(
        frame["LONGITUDE_PRED"] - frame["LONGITUDE_TRUE"],
        frame["LATITUDE_PRED"] - frame["LATITUDE_TRUE"],
    )
    frame["evaal_style_score"] = (
        frame["position_error"]
        + 50 * frame["BUILDINGID_PRED"].ne(frame["BUILDINGID"])
        + 4 * frame["FLOOR_PRED"].ne(frame["FLOOR"])
    )
    frame["floor_correct"] = frame["FLOOR_PRED"].eq(frame["FLOOR"])
    frame["building_correct"] = frame["BUILDINGID_PRED"].eq(frame["BUILDINGID"])
    return frame


def _load_m1_baseline(m3: pd.DataFrame) -> pd.DataFrame:
    records = pd.read_csv(ROOT / "results/phase2/m1/m1_all_runs.csv")
    records = records.loc[records.exp_id.eq("baseline")]
    rows = []
    for rec in records.itertuples(index=False):
        path = ROOT / rec.predictions_file
        m1 = pd.read_csv(path)
        # M1's CSV has local fold row order; M3 retains global source_row. Check
        # target/user metadata row-by-row before pairing the saved predictions.
        match = m3.loc[m3.fold.eq(rec.fold) & m3.seed.eq(rec.seed) & m3.exp_id.eq("a1_shared")].sort_values("source_row")
        m1 = m1.reset_index(drop=True)
        match = match.reset_index(drop=True)
        if len(m1) != len(match):
            raise ValueError(f"M1/M3 row count differs at fold={rec.fold}, seed={rec.seed}")
        for left, right in (("USERID", "USERID"), ("PHONEID", "PHONEID"),
                            ("BUILDINGID_TRUE", "BUILDINGID"), ("FLOOR_TRUE", "FLOOR")):
            if not np.array_equal(m1[left].to_numpy(), match[right].to_numpy()):
                raise ValueError(f"M1/M3 row order mismatch at fold={rec.fold}, seed={rec.seed}: {left}")
        for col in ("LONGITUDE_TRUE", "LATITUDE_TRUE"):
            # M3's exported truth coordinates are float32, whose spacing at
            # UTM northings is 0.5 m. Metadata and source-row order are checked
            # separately; this tolerance only accounts for that serialization.
            if not np.allclose(m1[col], match[col], atol=0.5, rtol=0):
                raise ValueError(f"M1/M3 coordinate order mismatch at fold={rec.fold}, seed={rec.seed}: {col}")
        frame = pd.DataFrame({
            "exp_id": "m1_baseline", "fold": rec.fold, "seed": rec.seed,
            "source_row": match.source_row, "USERID": m1.USERID, "PHONEID": m1.PHONEID,
            "BUILDINGID": m1.BUILDINGID_TRUE, "BUILDINGID_PRED": m1.BUILDINGID_PRED,
            "FLOOR": m1.FLOOR_TRUE, "FLOOR_PRED": m1.FLOOR_PRED,
            "LONGITUDE_TRUE": m1.LONGITUDE_TRUE, "LATITUDE_TRUE": m1.LATITUDE_TRUE,
            "LONGITUDE_PRED": m1.LONGITUDE_PRED, "LATITUDE_PRED": m1.LATITUDE_PRED,
        })
        rows.append(frame)
    baseline = pd.concat(rows, ignore_index=True)
    baseline["position_error"] = np.hypot(
        baseline.LONGITUDE_PRED - baseline.LONGITUDE_TRUE,
        baseline.LATITUDE_PRED - baseline.LATITUDE_TRUE,
    )
    baseline["evaal_style_score"] = (
        baseline.position_error
        + 50 * baseline.BUILDINGID_PRED.ne(baseline.BUILDINGID)
        + 4 * baseline.FLOOR_PRED.ne(baseline.FLOOR)
    )
    baseline["floor_correct"] = baseline.FLOOR_PRED.eq(baseline.FLOOR)
    baseline["building_correct"] = baseline.BUILDINGID_PRED.eq(baseline.BUILDINGID)
    return baseline


def _load_valid_m2_arms(m3: pd.DataFrame, clean_train: pd.DataFrame) -> pd.DataFrame:
    """Add completed M2 A4/A7 arms; intentionally omit stub-backed A8 runs."""
    archive = pd.read_csv(ROOT / "results/phase2/m2/m2_oof_predictions.csv.gz")
    parsed = archive.run.str.extract(r"^(?P<arm>.+)_f(?P<fold>\d+)_s(?P<seed>\d+)$")
    archive = archive.assign(arm=parsed.arm, fold=parsed.fold.astype(int), seed=parsed.seed.astype(int))
    arms = {"a4_linear": "m2_a4_linear", "a7_dae_pretrained": "m2_a7_dae_pretrained",
            "a7_random_init": "m2_a7_random_init"}
    parts = []
    for arm, output_name in arms.items():
        for (fold, seed), rows in archive.loc[archive.arm.eq(arm)].groupby(["fold", "seed"]):
            reference = m3.loc[m3.exp_id.eq("a1_shared") & m3.fold.eq(fold) & m3.seed.eq(seed)].sort_values("source_row").reset_index(drop=True)
            rows = rows.sort_values("index").reset_index(drop=True)
            if len(rows) != len(reference):
                raise ValueError(f"M2/M3 row count differs at {arm}, fold={fold}, seed={seed}")
            for left, right in (("USERID", "USERID"), ("PHONEID", "PHONEID"),
                                ("BUILDINGID_TRUE", "BUILDINGID"), ("FLOOR_TRUE", "FLOOR")):
                if not np.array_equal(rows[left], reference[right]):
                    raise ValueError(f"M2/M3 row order mismatch at {arm}, fold={fold}, seed={seed}: {left}")
            for col in ("LONGITUDE_TRUE", "LATITUDE_TRUE"):
                if not np.allclose(rows[col], reference[col], atol=0.5, rtol=0):
                    raise ValueError(f"M2/M3 coordinate order mismatch at {arm}, fold={fold}, seed={seed}: {col}")
            part = pd.DataFrame({
                "exp_id": output_name, "fold": fold, "seed": seed,
                "source_row": reference.source_row, "USERID": rows.USERID, "PHONEID": rows.PHONEID,
                "BUILDINGID": rows.BUILDINGID_TRUE, "BUILDINGID_PRED": rows.BUILDINGID_PRED,
                "FLOOR": rows.FLOOR_TRUE, "FLOOR_PRED": rows.FLOOR_PRED,
                "LONGITUDE_TRUE": clean_train.iloc[reference.source_row.to_numpy(dtype=int)].LONGITUDE.to_numpy(),
                "LATITUDE_TRUE": clean_train.iloc[reference.source_row.to_numpy(dtype=int)].LATITUDE.to_numpy(),
                "LONGITUDE_PRED": rows.LONGITUDE_PRED, "LATITUDE_PRED": rows.LATITUDE_PRED,
            })
            part["position_error"] = np.hypot(part.LONGITUDE_PRED - part.LONGITUDE_TRUE,
                                               part.LATITUDE_PRED - part.LATITUDE_TRUE)
            part["evaal_style_score"] = (part.position_error
                + 50 * part.BUILDINGID_PRED.ne(part.BUILDINGID)
                + 4 * part.FLOOR_PRED.ne(part.FLOOR))
            part["floor_correct"] = part.FLOOR_PRED.eq(part.FLOOR)
            part["building_correct"] = part.BUILDINGID_PRED.eq(part.BUILDINGID)
            parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _metric_table(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = ["evaal_style_score", "position_error", "floor_correct", "building_correct"]
    seed_summary = frame.groupby(["exp_id", "seed"], as_index=False).agg(
        n_scans=("source_row", "size"),
        evaal_style_score=("evaal_style_score", "mean"),
        mean_position_error=("position_error", "mean"),
        floor_accuracy=("floor_correct", "mean"),
        building_accuracy=("building_correct", "mean"),
    )
    per_user = frame.groupby(["exp_id", "USERID"], as_index=False).agg(
        n_scan_seed_pairs=("source_row", "size"),
        evaal_style_score=("evaal_style_score", "mean"),
        mean_position_error=("position_error", "mean"),
        floor_accuracy=("floor_correct", "mean"),
        building_accuracy=("building_correct", "mean"),
    )
    return seed_summary, per_user


def _paired_user_bootstrap(per_user: pd.DataFrame, *, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    metric_names = ["evaal_style_score", "mean_position_error", "floor_accuracy", "building_accuracy"]
    models = sorted(per_user.exp_id.unique())
    users = np.array(sorted(per_user.USERID.unique()))
    indexed = per_user.set_index(["exp_id", "USERID"])
    rows = []
    for model in models:
        for reference in models:
            if model == reference:
                continue
            for metric in metric_names:
                differences = np.array([
                    indexed.loc[(model, uid), metric] - indexed.loc[(reference, uid), metric]
                    for uid in users
                ], dtype=float)
                draws = rng.integers(0, len(users), size=(n_boot, len(users)))
                boot = differences[draws].mean(axis=1)
                rows.append({
                    "model": model, "reference": reference, "metric": metric,
                    "difference_model_minus_reference": float(differences.mean()),
                    "ci95_low": float(np.quantile(boot, 0.025)),
                    "ci95_high": float(np.quantile(boot, 0.975)),
                    "n_users": len(users), "bootstrap_replicates": n_boot,
                    "unit": "USERID cluster; per-user average over scans and seeds",
                })
    return pd.DataFrame(rows)


def _mcnemar_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    seed_frame = frame.loc[frame.seed.eq(42)]
    models = sorted(seed_frame.exp_id.unique())
    for model in models:
        for reference in models:
            if model >= reference:
                continue
            left = seed_frame.loc[seed_frame.exp_id.eq(model)].sort_values("source_row")
            right = seed_frame.loc[seed_frame.exp_id.eq(reference)].sort_values("source_row")
            if not np.array_equal(left.source_row.to_numpy(), right.source_row.to_numpy()):
                raise ValueError(f"Unmatched OOF rows for {model} vs {reference}")
            left_ok, right_ok = left.floor_correct.to_numpy(), right.floor_correct.to_numpy()
            b = int(np.sum(left_ok & ~right_ok))
            c = int(np.sum(~left_ok & right_ok))
            p = float(binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue) if b + c else 1.0
            rows.append({"model": model, "reference": reference, "seed": 42,
                         "model_only_correct": b, "reference_only_correct": c,
                         "exact_mcnemar_p_scan_level": p,
                         "caveat": "scan-level p-value ignores within-USERID dependence; interpret with cluster-bootstrap CI"})
    return pd.DataFrame(rows)


def _expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for i in range(bins):
        in_bin = (confidence >= edges[i]) & (confidence < edges[i + 1])
        if i == bins - 1:
            in_bin |= confidence == 1.0
        if in_bin.any():
            result += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(result)


def _floor_calibration(m3: pd.DataFrame, clean_train: pd.DataFrame) -> pd.DataFrame:
    """Infer held-out floor probabilities from M3 checkpoints for calibration."""
    rows = []
    for exp_id in M3_ARMS:
        for seed in (42, 1337, 2024):
            truths: list[np.ndarray] = []
            probabilities: list[np.ndarray] = []
            for fold in range(5):
                checkpoint_path = ROOT / "results/phase2/m3/checkpoints" / f"model_{exp_id}_f{fold}_s{seed}.pt"
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                held_out = m3.loc[m3.exp_id.eq(exp_id) & m3.fold.eq(fold) & m3.seed.eq(seed)].sort_values("source_row")
                source_ids = held_out.source_row.to_numpy(dtype=int)
                dataset = Phase2Dataset(
                    clean_train.iloc[source_ids],
                    np.asarray(checkpoint["coordinate_mean"], dtype=np.float32),
                    np.asarray(checkpoint["coordinate_scale"], dtype=np.float32),
                    include_missingness=(int(checkpoint["input_dim"]) == 1040),
                )
                model = make_model(**checkpoint["config"])
                model.load_state_dict(checkpoint["state_dict"])
                model.eval()
                prob_parts, target_parts = [], []
                with torch.no_grad():
                    for batch in DataLoader(dataset, batch_size=512, shuffle=False):
                        logits = model(batch["features"])["floor_logits"]
                        prob_parts.append(torch.softmax(logits, dim=1).numpy())
                        target_parts.append(batch["floor"].numpy())
                probabilities.append(np.concatenate(prob_parts))
                truths.append(np.concatenate(target_parts))
            y = np.concatenate(truths)
            p = np.concatenate(probabilities)
            confidence = p.max(axis=1)
            predicted = p.argmax(axis=1)
            correct = predicted == y
            one_hot = np.eye(p.shape[1])[y]
            rows.append({
                "exp_id": exp_id, "seed": seed, "n_scans": len(y), "ece_15bin": _expected_calibration_error(confidence, correct),
                "multiclass_brier": float(np.mean(np.sum((p - one_hot) ** 2, axis=1))),
                "negative_log_likelihood": float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0)))),
                "floor_accuracy": float(correct.mean()),
                "calibration_method": "top-label ECE, 15 equal-width confidence bins",
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261115)
    args = parser.parse_args()
    out = ROOT / "results/phase2/m4"
    out.mkdir(parents=True, exist_ok=True)

    clean_train, _, _ = load_data()
    m3 = _prepare_m3(ROOT / "results/phase2/m3/m3_oof_predictions.csv.gz", clean_train)
    m3 = m3.loc[m3.exp_id.isin(M3_ARMS)].copy()
    baseline = _load_m1_baseline(m3)
    m2 = _load_valid_m2_arms(m3, clean_train)
    frame = pd.concat([m3, baseline, m2], ignore_index=True)
    seed_summary, per_user = _metric_table(frame)
    seed_summary.to_csv(out / "m4_seed_metrics.csv", index=False)
    per_user.to_csv(out / "m4_per_user_metrics.csv", index=False)
    _paired_user_bootstrap(per_user, n_boot=args.bootstrap, seed=args.seed).to_csv(
        out / "m4_paired_user_bootstrap.csv", index=False
    )
    _mcnemar_diagnostics(frame).to_csv(out / "m4_floor_mcnemar_seed42.csv", index=False)
    _floor_calibration(m3, clean_train).to_csv(out / "m4_floor_calibration.csv", index=False)

    best = frame.loc[frame.exp_id.eq("a1_independent")].copy()
    best["error_rank"] = best.groupby("seed")["position_error"].rank(method="first", ascending=False)
    worst = best.loc[best.error_rank.le(100)].sort_values(["seed", "error_rank"])
    worst.to_csv(out / "m4_worst_100_scans_per_seed.csv", index=False)
    print(f"Saved paired M4 OOF analysis for {frame.USERID.nunique()} users to {out}")
    print("Note: M1 baseline predictions are paired by fold/seed after strict row-order checks.")
    print("Floor calibration metrics were inferred from saved M3 fold checkpoints.")
    print("Learning curves are not included yet; M4 still needs the fixed-protocol 25/50/75/100% retrains.")


if __name__ == "__main__":
    main()
