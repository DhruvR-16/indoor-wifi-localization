"""A8 hybrid: multitask-trunk embedding + routed local coordinate estimator.

STATUS: the pipeline is complete, but its numbers are PRELIMINARY until M3
delivers the real multitask model. Until then the embedding comes from
`src/models/stub_multitask.py`, every record carries ``"blocked_on_m3": true``,
and a stub-backed number is not an A8 result - in particular it must never be
called "the multitask embedding".

Candidates, all scored on identical held-out rows of the same fold and seed:

* ``a8_mlp_head``        - the multitask coordinate head (the DL baseline).
* ``a8_embed_knn``       - distance-weighted kNN on the 128-D embedding, routed
                           by the model's own predicted building/floor.
* ``a8_embed_xgb``       - XGBoost on the same embedding and route, testing
                           whether a gain is specific to local distance methods.
* ``a8_embed_knn_oracle``- DIAGNOSTIC ONLY: the same kNN routed by true labels.
                           Not end-to-end performance; excluded from headlines.

Example
-------
python scripts/run_m2_a8_hybrid.py --folds 0 --seeds 42
"""

from __future__ import annotations

# macOS OpenMP guard - must run before torch or xgboost is imported. torch
# ships its own OpenMP runtime and xgboost links Homebrew's libomp; loading
# both multi-threaded in one process segfaults the interpreter. Single-threaded
# OpenMP is the only workaround that held up (KMP_DUPLICATE_LIB_OK did not).
import os

for _omp_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "XGBOOST_NUM_THREADS"):
    os.environ.setdefault(_omp_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.phase2_m2 import make_loader  # noqa: E402
from src.evaluation.metrics import evaluate_predictions  # noqa: E402
from src.models.hybrid import (  # noqa: E402
    DEFAULT_KNN_GRID,
    extract_embeddings,
    fit_routed_estimator,
    tune_routed_estimator,
)
from src.models.ml.regressors import make_base_regressor  # noqa: E402
from src.training.m2_trainer import M2TrainConfig, train_fold_models  # noqa: E402
from src.training.phase2_trainer import compute_metrics_and_predictions  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "phase2_m2.yaml"
RESULTS_DIR = ROOT / "results" / "phase2" / "m2"
MODEL_SOURCE = "src/models/stub_multitask.py (M3 multitask model not yet delivered)"


def _predictions_frame(
    base_frame: pd.DataFrame,
    coordinates_norm: np.ndarray,
    coord_scaler,
) -> pd.DataFrame:
    """Rebuild a predictions frame with new coordinates, keeping labels and row ids.

    The hybrid replaces only the coordinate estimate; building and floor still
    come from the multitask classification heads, so those columns are carried
    over unchanged and the rows stay aligned with every other candidate.
    """
    frame = base_frame.copy()
    predicted_m = coord_scaler.inverse_transform(coordinates_norm)
    frame["LONGITUDE_PRED"] = predicted_m[:, 0]
    frame["LATITUDE_PRED"] = predicted_m[:, 1]
    frame["position_error"] = np.hypot(
        frame["LONGITUDE_PRED"] - frame["LONGITUDE_TRUE"],
        frame["LATITUDE_PRED"] - frame["LATITUDE_TRUE"],
    )
    return frame


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    reference = config["reference"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", nargs="+", type=int, default=config["folds"])
    parser.add_argument("--seeds", nargs="+", type=int, default=config["seeds"])
    parser.add_argument("--representation", default="linear")
    parser.add_argument(
        "--route-mode",
        default="per_building_floor",
        choices=["global", "per_building", "per_building_floor"],
    )
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--max-epochs", type=int, default=reference["max_epochs"])
    parser.add_argument("--patience", type=int, default=reference["patience"])
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    (results_dir / "preds").mkdir(parents=True, exist_ok=True)
    (results_dir / "records").mkdir(parents=True, exist_ok=True)

    print(
        f"A8 hybrid (PRELIMINARY - embedding source: {MODEL_SOURCE})\n"
        f"  {len(args.folds)} folds x {len(args.seeds)} seeds | route: {args.route_mode} "
        f"| kNN grid: {len(DEFAULT_KNN_GRID)} settings",
        flush=True,
    )
    started = time.time()

    for fold in args.folds:
        for seed in args.seeds:
            marker = (
                results_dir / "records" / f"record_a8_embed_knn_f{fold}_s{seed}.json"
            )
            if args.skip_existing and marker.exists():
                print(f"  f{fold} s{seed}: existing records, skipped", flush=True)
                continue

            run_started = time.time()
            train_config = M2TrainConfig(
                exp_id="a8_backbone",
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
                notes={"ablation": "A8", "factor": "coordinate_estimator"},
            )
            trained = train_fold_models(train_config)

            # The inner model never saw the inner-validation rows, so the
            # estimator settings are not chosen on rows its encoder was fit on.
            inner_train_emb = extract_embeddings(
                trained.inner_model, trained.inner_train_ds, trained.device
            )
            inner_val_emb = extract_embeddings(
                trained.inner_model, trained.inner_val_ds, trained.device
            )
            selection = tune_routed_estimator(
                inner_train=inner_train_emb,
                inner_val=inner_val_emb,
                coord_scaler=trained.coord_scaler,
                estimator_name="knn",
                route_mode=args.route_mode,
                grid=DEFAULT_KNN_GRID,
                seed=seed,
                min_samples=args.min_samples,
                route_with_predictions=True,
            )

            outer_train_emb = extract_embeddings(
                trained.outer_model, trained.outer_train_ds, trained.device
            )
            outer_val_emb = extract_embeddings(
                trained.outer_model, trained.outer_val_ds, trained.device
            )

            outer_val_loader = make_loader(
                trained.outer_val_ds, train_config.batch_size, False, seed
            )
            head_metrics, head_preds = compute_metrics_and_predictions(
                trained.outer_model,
                outer_val_loader,
                trained.coord_scaler,
                trained.device,
            )
            assert np.array_equal(
                head_preds["index"].to_numpy(), outer_val_emb["index"]
            ), "embedding rows are not aligned with the head's prediction rows"

            candidates: dict[str, tuple[pd.DataFrame, dict]] = {
                "a8_mlp_head": (head_preds, {"estimator": "multitask_coordinate_head"})
            }

            for exp_id, estimator_name, params, routed_by in (
                ("a8_embed_knn", "knn", selection.params, "predicted"),
                ("a8_embed_xgb", "xgboost", {"n_jobs": 1}, "predicted"),
                ("a8_embed_knn_oracle", "knn", selection.params, "oracle"),
            ):
                estimator = make_base_regressor(estimator_name, seed=seed, **params)
                routed = fit_routed_estimator(
                    embeddings=outer_train_emb["embedding"],
                    coordinates_norm=outer_train_emb["coordinates_norm"],
                    buildings=outer_train_emb["building_true"],
                    floors=outer_train_emb["floor_true"],
                    estimator=estimator,
                    route_mode=args.route_mode,
                    min_samples=args.min_samples,
                )
                key = "true" if routed_by == "oracle" else "pred"
                predicted_norm = routed.predict(
                    outer_val_emb["embedding"],
                    route_buildings=outer_val_emb[f"building_{key}"],
                    route_floors=outer_val_emb[f"floor_{key}"],
                )
                frame = _predictions_frame(
                    head_preds, predicted_norm, trained.coord_scaler
                )
                candidates[exp_id] = (
                    frame,
                    {
                        "estimator": estimator_name,
                        "params": params,
                        "routed_by": routed_by,
                        "route_mode": args.route_mode,
                        "diagnostic_only": routed_by == "oracle",
                    },
                )

            summary_line = []
            for exp_id, (frame, estimator_info) in candidates.items():
                metrics = evaluate_predictions(frame)
                run_name = f"{exp_id}_f{fold}_s{seed}"
                preds_file = results_dir / "preds" / f"preds_{run_name}.csv"
                frame.to_csv(preds_file, index=False)

                record = {
                    "member": "M2",
                    "exp_id": exp_id,
                    "fold": fold,
                    "seed": seed,
                    "representation": args.representation,
                    "include_missing_mask": False,
                    "input_dim": trained.input_dim,
                    "embedding_dim": int(outer_val_emb["embedding"].shape[1]),
                    "best_epoch": trained.best_epoch,
                    "best_inner_evaal": trained.best_inner_evaal,
                    "final_train_loss": trained.epoch_curves[-1]["train_loss"],
                    "final_val_loss": trained.epoch_curves[-1]["val_loss"],
                    "metrics": metrics,
                    "config": asdict(train_config),
                    "estimator": estimator_info,
                    "tuning": (
                        asdict(selection)
                        if estimator_info["estimator"] == "knn"
                        else None
                    ),
                    "pretraining": trained.transfer_info,
                    "inner_split": asdict(trained.inner_report),
                    "blocked_on_m3": True,
                    "model_source": MODEL_SOURCE,
                    "preliminary": (
                        "Embedding comes from the M3 interface stub, not M3's multitask "
                        "model. Not reportable as an A8 result."
                    ),
                    "software": {
                        "torch": torch.__version__,
                        "sklearn": sklearn.__version__,
                        "numpy": np.__version__,
                        "python": platform.python_version(),
                        "device": str(trained.device),
                    },
                    "meta": trained.meta,
                    "predictions_file": str(preds_file),
                }
                with open(
                    results_dir / "records" / f"record_{run_name}.json", "w"
                ) as f:
                    json.dump(record, f, indent=2, default=str)

                summary_line.append(
                    f"{exp_id.replace('a8_',''):>16} "
                    f"mean {metrics['mean_position_error']:6.2f} m "
                    f"EvAAL {metrics['evaal_style_score']:6.2f} m"
                )

            print(
                f"  f{fold} s{seed}: E*={trained.best_epoch} "
                f"k={selection.params.get('n_neighbors')} p={selection.params.get('p')} "
                f"(inner {selection.inner_mean_position_error_m:.2f} m) "
                f"[{time.time()-run_started:.0f}s]",
                flush=True,
            )
            for line in summary_line:
                print(f"      {line}", flush=True)

    print(f"\nTotal wall time: {(time.time()-started)/60:.1f} min")
    print("REMINDER: A8 numbers stay preliminary until M3's multitask model lands.")


if __name__ == "__main__":
    main()
