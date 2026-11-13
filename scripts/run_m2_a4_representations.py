"""A4: compare RSSI input representations on matched folds, seeds and capacity.

Each arm changes one factor - the input view - and inherits everything else
from `configs/phase2_m2.yaml:reference`. Run records land in
results/phase2/m2/records/ and row-level out-of-fold predictions in
results/phase2/m2/preds/.

Examples
--------
# smoke: reference arm, one fold, one seed
python scripts/run_m2_a4_representations.py --arms a4_linear --folds 0 --seeds 42

# full A4 grid (6 arms x 5 folds x 3 seeds)
python scripts/run_m2_a4_representations.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.m2_trainer import M2TrainConfig, run_m2_fold  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "phase2_m2.yaml"
RESULTS_DIR = ROOT / "results" / "phase2" / "m2"


def parse_args() -> argparse.Namespace:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    arm_ids = [arm["exp_id"] for arm in config["a4_arms"]]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=arm_ids, choices=arm_ids)
    parser.add_argument("--folds", nargs="+", type=int, default=config["folds"])
    parser.add_argument("--seeds", nargs="+", type=int, default=config["seeds"])
    parser.add_argument(
        "--inner-split",
        default=config["reference"]["inner_split_mode"],
        choices=["stratified", "legacy"],
    )
    parser.add_argument(
        "--max-epochs", type=int, default=config["reference"]["max_epochs"]
    )
    parser.add_argument("--patience", type=int, default=config["reference"]["patience"])
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Leave runs that already have a record on disk untouched.",
    )
    args = parser.parse_args()
    args.config = config
    return args


def main() -> None:
    args = parse_args()
    config = args.config
    reference = config["reference"]
    representations = config["representations"]
    arms = {arm["exp_id"]: arm for arm in config["a4_arms"]}
    results_dir = Path(args.results_dir)
    records_dir = results_dir / "records"

    jobs = [
        (arm_id, fold, seed)
        for arm_id in args.arms
        for fold in args.folds
        for seed in args.seeds
    ]
    print(
        f"A4: {len(args.arms)} arms x {len(args.folds)} folds x {len(args.seeds)} seeds "
        f"= {len(jobs)} runs | inner split: {args.inner_split}",
        flush=True,
    )

    rows: list[dict] = []
    started = time.time()
    for position, (arm_id, fold, seed) in enumerate(jobs, start=1):
        arm = arms[arm_id]
        record_file = records_dir / f"record_{arm_id}_f{fold}_s{seed}.json"
        if args.skip_existing and record_file.exists():
            print(
                f"[{position}/{len(jobs)}] {arm_id} f{fold} s{seed}: existing record, skipped",
                flush=True,
            )
            continue

        train_config = M2TrainConfig(
            exp_id=arm_id,
            fold=fold,
            seed=seed,
            representation=arm["representation"],
            include_missing_mask=arm["include_missing_mask"],
            alpha=representations["alpha"],
            beta=representations["beta"],
            hidden_dims=tuple(reference["hidden_dims"]),
            dropout=reference["dropout"],
            batch_norm=reference["batch_norm"],
            weight_decay=reference["weight_decay"],
            lr=reference["lr"],
            batch_size=reference["batch_size"],
            max_epochs=args.max_epochs,
            patience=args.patience,
            noise_sigma=reference["noise_sigma"],
            ap_dropout=reference["ap_dropout"],
            inner_split_mode=args.inner_split,
            inner_splits=reference["inner_splits"],
            loss_weights=tuple(reference["loss_weights"]),
            notes={"ablation": "A4", "factor": "rssi_representation"},
            **({"device": args.device} if args.device else {}),
        )

        run_started = time.time()
        record = run_m2_fold(train_config, results_dir=results_dir)
        metrics = record["metrics"]
        rows.append(
            {
                "exp_id": arm_id,
                "fold": fold,
                "seed": seed,
                "representation": record["representation"],
                "include_missing_mask": record["include_missing_mask"],
                "input_dim": record["input_dim"],
                "best_epoch": record["best_epoch"],
                "best_inner_evaal": record["best_inner_evaal"],
                **{k: v for k, v in metrics.items()},
            }
        )
        print(
            f"[{position}/{len(jobs)}] {arm_id} f{fold} s{seed}: "
            f"E*={record['best_epoch']:>2} "
            f"bldg {metrics['building_accuracy']*100:5.2f}% "
            f"floor {metrics['floor_accuracy']*100:5.2f}% "
            f"mean {metrics['mean_position_error']:6.2f} m "
            f"EvAAL {metrics['evaal_style_score']:6.2f} m "
            f"({time.time()-run_started:.0f}s)",
            flush=True,
        )

    if rows:
        out_file = results_dir / "m2_a4_runs.csv"
        frame = pd.DataFrame(rows)
        if out_file.exists():
            previous = pd.read_csv(out_file)
            frame = (
                pd.concat([previous, frame], ignore_index=True)
                .drop_duplicates(subset=["exp_id", "fold", "seed"], keep="last")
                .sort_values(["exp_id", "fold", "seed"])
            )
        frame.to_csv(out_file, index=False)
        print(f"\nWrote {len(frame)} A4 run rows to {out_file}")
    print(f"Total wall time: {(time.time()-started)/60:.1f} min")


if __name__ == "__main__":
    main()
