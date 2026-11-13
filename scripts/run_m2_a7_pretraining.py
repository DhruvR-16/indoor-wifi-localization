"""A7: random initialisation vs fold-matched DAE pretraining, architecture-matched.

Both arms train the same 520 -> 256 -> 128 multitask trunk on the same folds,
seeds, optimiser and stopping rule. The only difference is the trunk's starting
weights:

* ``a7_random_init``      - standard initialisation (protocol-identical to the
                            A4 reference arm ``a4_linear``).
* ``a7_dae_pretrained``   - the encoder from ``dae_<rep>_f<fold>_s<seed>.pt``,
                            trained only on that fold's outer-training users.

The checkpoint's fold and input width are verified before transfer, and a
shape mismatch raises instead of skipping or reshaping layers, so the two arms
can never silently differ by architecture.

Run `scripts/train_m2_dae.py` first.

Example
-------
python scripts/run_m2_a7_pretraining.py --folds 0 1 2 3 4 --seeds 42 1337 2024
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
DAE_DIR = RESULTS_DIR / "dae"


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    arm_ids = [arm["exp_id"] for arm in config["a7_arms"]]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=arm_ids, choices=arm_ids)
    parser.add_argument("--folds", nargs="+", type=int, default=config["folds"])
    parser.add_argument("--seeds", nargs="+", type=int, default=config["seeds"])
    parser.add_argument("--representation", default="linear")
    parser.add_argument("--dae-dir", default=str(DAE_DIR))
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument(
        "--max-epochs", type=int, default=config["reference"]["max_epochs"]
    )
    parser.add_argument("--patience", type=int, default=config["reference"]["patience"])
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    reference = config["reference"]
    arms = {arm["exp_id"]: arm for arm in config["a7_arms"]}
    results_dir = Path(args.results_dir)
    dae_dir = Path(args.dae_dir)

    jobs = [
        (arm_id, fold, seed)
        for arm_id in args.arms
        for fold in args.folds
        for seed in args.seeds
    ]
    print(
        f"A7: {len(args.arms)} arms x {len(args.folds)} folds x {len(args.seeds)} seeds "
        f"= {len(jobs)} runs | view: {args.representation}",
        flush=True,
    )

    rows: list[dict] = []
    started = time.time()
    for position, (arm_id, fold, seed) in enumerate(jobs, start=1):
        record_file = results_dir / "records" / f"record_{arm_id}_f{fold}_s{seed}.json"
        if args.skip_existing and record_file.exists():
            print(
                f"[{position}/{len(jobs)}] {arm_id} f{fold} s{seed}: existing record, skipped",
                flush=True,
            )
            continue

        pretrained_path = None
        if arms[arm_id]["pretrained"]:
            pretrained_path = dae_dir / f"dae_{args.representation}_f{fold}_s{seed}.pt"
            if not pretrained_path.exists():
                raise FileNotFoundError(
                    f"Missing fold-matched DAE checkpoint {pretrained_path}. "
                    "Run scripts/train_m2_dae.py first."
                )

        train_config = M2TrainConfig(
            exp_id=arm_id,
            fold=fold,
            seed=seed,
            representation=args.representation,
            include_missing_mask=False,
            alpha=config["representations"]["alpha"],
            beta=config["representations"]["beta"],
            hidden_dims=tuple(reference["hidden_dims"]),
            dropout=reference["dropout"],
            batch_norm=reference["batch_norm"],
            weight_decay=reference["weight_decay"],
            lr=reference["lr"],
            batch_size=reference["batch_size"],
            max_epochs=args.max_epochs,
            patience=args.patience,
            inner_split_mode=reference["inner_split_mode"],
            inner_splits=reference["inner_splits"],
            loss_weights=tuple(reference["loss_weights"]),
            pretrained_encoder=str(pretrained_path) if pretrained_path else None,
            notes={"ablation": "A7", "factor": "trunk_initialisation"},
        )

        run_started = time.time()
        record = run_m2_fold(train_config, results_dir=results_dir)
        metrics = record["metrics"]
        rows.append(
            {
                "exp_id": arm_id,
                "fold": fold,
                "seed": seed,
                "pretrained": record["pretraining"]["pretrained"],
                "transferred_tensors": record["pretraining"].get(
                    "transferred_tensors", 0
                ),
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
        out_file = results_dir / "m2_a7_runs.csv"
        frame = pd.DataFrame(rows)
        if out_file.exists():
            previous = pd.read_csv(out_file)
            frame = (
                pd.concat([previous, frame], ignore_index=True)
                .drop_duplicates(subset=["exp_id", "fold", "seed"], keep="last")
                .sort_values(["exp_id", "fold", "seed"])
            )
        frame.to_csv(out_file, index=False)
        print(f"\nWrote {len(frame)} A7 run rows to {out_file}")
    print(f"Total wall time: {(time.time()-started)/60:.1f} min")


if __name__ == "__main__":
    main()
