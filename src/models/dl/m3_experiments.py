"""Leakage-safe M3 A1/A2/A3 experiment runner for saved USERID outer folds."""
from __future__ import annotations

import copy
import json
import platform

import sklearn
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score

from src.data.load import load_data
from src.data.phase2_m3 import make_phase2_dataloaders, move_batch
from src.evaluation.metrics import evaluate_predictions
from src.models.dl.multitask import make_model, multitask_loss
from src.models.dl.m3_support import prediction_frame, select_device


@dataclass(frozen=True)
class M3RunOptions:
    epochs: int = 200
    patience: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    device: str = "auto"
    seeds: tuple[int, ...] = (42, 1337, 2024)


EXPERIMENTS: dict[str, dict[str, Any]] = {
    "a1_shared": {},
    "a1_independent": {"independent_tasks": True},
    "a2_equal_weights": {},
    "a2_inner_selected_weights": {"select_weights_on_inner": True},
    "a2_uncertainty_weights": {"loss_weighting": "uncertainty"},
    "a3_flat_floor": {},
    "a3_hierarchical_floor": {"floor_mode": "hierarchical"},
}
# Shared/flat/equal is one identical control and is recorded once as a1_shared.
DEFAULT_EXPERIMENTS = ("a1_shared", "a1_independent", "a2_inner_selected_weights",
                       "a2_uncertainty_weights", "a3_hierarchical_floor")
WEIGHT_CANDIDATES = ((0.5, 0.5, 2.0), (1.0, 1.0, 1.0), (1.0, 1.0, 2.0))


def _inner_partition(frame: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Pick a GroupKFold inner partition with every validation building in training."""
    groups = frame["USERID"].to_numpy()
    buildings = frame["BUILDINGID"].to_numpy()
    candidates = []
    n_groups = len(np.unique(groups))
    splitter = GroupKFold(n_splits=min(4, n_groups))
    for train_idx, valid_idx in splitter.split(frame, groups=groups):
        missing = sorted(set(buildings[valid_idx]) - set(buildings[train_idx]))
        ratio = len(valid_idx) / len(frame)
        candidates.append((len(missing), abs(ratio - 0.25), train_idx, valid_idx, missing))
    viable = [c for c in candidates if not c[0]]
    if not viable:
        raise ValueError("No USERID-grouped inner split covers all validation building labels")
    _, _, train_idx, valid_idx, _ = min(viable, key=lambda item: (item[0], item[1]))
    info = {"method": "GroupKFold", "n_splits": min(4, n_groups),
            "train_users": sorted(int(x) for x in np.unique(groups[train_idx])),
            "valid_users": sorted(int(x) for x in np.unique(groups[valid_idx])),
            "train_rows": int(len(train_idx)), "valid_rows": int(len(valid_idx)),
            "missing_buildings_from_inner_train": []}
    return train_idx, valid_idx, info


def _fit_inner(factory_kwargs: dict[str, Any], train_frame: pd.DataFrame, valid_frame: pd.DataFrame,
               run_options: M3RunOptions, seed: int, device: torch.device,
               loss_weights: tuple[float, float, float], loss_weighting: str) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loader, valid_loader, mean, scale = make_phase2_dataloaders(
        train_frame, valid_frame, batch_size=run_options.batch_size, seed=seed,
        include_missingness=(factory_kwargs["input_dim"] == 1040))
    model = make_model(seed=seed, **factory_kwargs).to(device)
    optimizer = AdamW(model.parameters(), lr=run_options.learning_rate,
                      weight_decay=run_options.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
    curves: list[dict[str, float | int]] = []
    best_score, best_epoch, wait = float("inf"), 0, 0
    best_state = None
    for epoch in range(1, run_options.epochs + 1):
        model.train()
        train_total = 0.0
        train_components = np.zeros(3, dtype=float)
        train_rows = 0
        for source in train_loader:
            batch = move_batch(source, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["features"])
            loss, components = multitask_loss(outputs, batch, model, loss_weights)
            loss.backward()
            optimizer.step()
            n = len(batch["building"])
            train_total += float(loss.detach()) * n
            train_components += np.asarray([float(components[k].detach()) for k in ("building", "floor", "coordinates")]) * n
            train_rows += n
        model.eval()
        with torch.no_grad():
            valid_predictions = []
            valid_loss_sum = 0.0
            valid_components = np.zeros(3, dtype=float)
            valid_rows = 0
            for source in valid_loader:
                batch = move_batch(source, device)
                outputs = model(batch["features"])
                loss, components = multitask_loss(outputs, batch, model, loss_weights)
                n = len(batch["building"])
                valid_loss_sum += float(loss) * n
                valid_components += np.asarray([float(components[k]) for k in ("building", "floor", "coordinates")]) * n
                valid_rows += n
                valid_predictions.append(prediction_frame(outputs, batch, mean, scale))
        predictions = pd.concat(valid_predictions, ignore_index=True)
        metrics = evaluate_predictions(predictions)
        score = float(metrics["evaal_style_score"])
        scheduler.step(score)
        curves.append({"epoch": epoch, "train_loss": train_total / max(train_rows, 1),
                       "train_building_ce": train_components[0] / max(train_rows, 1),
                       "train_floor_ce": train_components[1] / max(train_rows, 1),
                       "train_coordinate_huber": train_components[2] / max(train_rows, 1),
                       "valid_loss": valid_loss_sum / max(valid_rows, 1),
                       "valid_building_ce": valid_components[0] / max(valid_rows, 1),
                       "valid_floor_ce": valid_components[1] / max(valid_rows, 1),
                       "valid_coordinate_huber": valid_components[2] / max(valid_rows, 1),
                       "valid_evaal": score,
                       "learning_rate": optimizer.param_groups[0]["lr"]})
        if score < best_score:
            best_score, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
        if wait >= run_options.patience:
            break
    if best_state is None:
        raise RuntimeError("Inner training produced no checkpoint")
    return {"best_score": best_score, "best_epoch": best_epoch,
            "state_dict": best_state, "curves": curves,
            "coordinate_mean": mean, "coordinate_scale": scale}


def run_m3_fold(exp_id: str, fold_id: int, train_frame: pd.DataFrame,
                train_indices: list[int], valid_indices: list[int], *,
                run_options: M3RunOptions, hidden_dims: tuple[int, ...] = (512, 256, 128),
                dropout: float = 0.3, batch_norm: bool = True,
                include_missingness: bool = False,
                output_dir: str | Path = "results/phase2/m3") -> dict[str, Any]:
    if exp_id not in EXPERIMENTS:
        raise ValueError(f"Unknown M3 experiment: {exp_id}")
    seed = run_options.seeds[0]
    # Seed is overridden by the caller via a one-seed options instance.
    outer_train = train_frame.iloc[train_indices].copy().reset_index(drop=True)
    outer_valid = train_frame.iloc[valid_indices].copy().reset_index(drop=True)
    inner_train_i, inner_valid_i, split_info = _inner_partition(outer_train, seed)
    inner_train = outer_train.iloc[inner_train_i].copy().reset_index(drop=True)
    inner_valid = outer_train.iloc[inner_valid_i].copy().reset_index(drop=True)
    arm = EXPERIMENTS[exp_id]
    factory_kwargs = {"input_dim": 1040 if include_missingness else 520,
                      "hidden_dims": hidden_dims, "dropout": dropout,
                      "batch_norm": batch_norm, **{k: v for k, v in arm.items()
                      if k in {"floor_mode", "loss_weighting", "independent_tasks"}}}
    loss_weighting = arm.get("loss_weighting", "fixed")
    candidates = WEIGHT_CANDIDATES if arm.get("select_weights_on_inner") else ((1., 1., 1.),)
    candidate_runs = []
    for weights in candidates:
        result = _fit_inner(factory_kwargs, inner_train, inner_valid, run_options,
                            seed, select_device(run_options.device), weights, loss_weighting)
        candidate_runs.append({"weights": weights, **result})
    selected = min(candidate_runs, key=lambda result: result["best_score"])
    selected_weights = tuple(float(x) for x in selected["weights"])
    best_epoch = int(selected["best_epoch"])

    # Refit from a fresh initialization on all outer-training users. Outer labels
    # are not used for stopping, weight selection, or preprocessing.
    seed = run_options.seeds[0]
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = select_device(run_options.device)
    outer_train_loader, outer_valid_loader, mean, scale = make_phase2_dataloaders(
        outer_train, outer_valid, batch_size=run_options.batch_size, seed=seed,
        include_missingness=include_missingness)
    model = make_model(seed=seed, **factory_kwargs).to(device)
    optimizer = AdamW(model.parameters(), lr=run_options.learning_rate,
                      weight_decay=run_options.weight_decay)
    outer_curves = []
    for epoch in range(1, best_epoch + 1):
        model.train()
        total, rows = 0.0, 0
        task_totals = np.zeros(3, dtype=float)
        for source in outer_train_loader:
            batch = move_batch(source, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["features"])
            loss, components = multitask_loss(outputs, batch, model, selected_weights)
            loss.backward()
            optimizer.step()
            n = len(batch["building"])
            total += float(loss.detach()) * n
            task_totals += np.asarray([float(components[k].detach()) for k in ("building", "floor", "coordinates")]) * n
            rows += n
        outer_curves.append({"epoch": epoch, "train_loss": total / max(rows, 1),
                             "train_building_ce": task_totals[0] / max(rows, 1),
                             "train_floor_ce": task_totals[1] / max(rows, 1),
                             "train_coordinate_huber": task_totals[2] / max(rows, 1)})
    model.eval()
    with torch.no_grad():
        batches = [move_batch(b, device) for b in outer_valid_loader]
        outputs = [model(b["features"]) for b in batches]
        predictions = pd.concat([prediction_frame(o, b, mean, scale)
                                  for o, b in zip(outputs, batches)], ignore_index=True)
    # Keep original cleaned-train row id and held-out user/phone identity.
    row_frame = predictions.copy()
    row_frame.insert(0, "source_row", valid_indices)
    row_frame.insert(1, "USERID", outer_valid["USERID"].to_numpy())
    row_frame.insert(2, "PHONEID", outer_valid["PHONEID"].to_numpy())
    metrics = evaluate_predictions(row_frame)
    metrics["building_macro_f1"] = float(f1_score(row_frame["BUILDINGID"], row_frame["BUILDINGID_PRED"], average="macro", zero_division=0))
    metrics["floor_macro_f1"] = float(f1_score(row_frame["FLOOR"], row_frame["FLOOR_PRED"], average="macro", zero_division=0))
    out = Path(output_dir)
    for sub in ("preds", "checkpoints", "records", "curves"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    run = f"{exp_id}_f{fold_id}_s{run_options.seeds[0]}"
    pred_path = out / "preds" / f"preds_{run}.csv"
    row_frame.to_csv(pred_path, index=False)
    checkpoint_path = out / "checkpoints" / f"model_{run}.pt"
    torch.save({"state_dict": model.cpu().state_dict(), "config": factory_kwargs,
                "loss_weights": selected_weights, "loss_weighting": loss_weighting,
                "fold": fold_id, "seed": run_options.seeds[0], "best_epoch": best_epoch,
                "coordinate_mean": mean.tolist(), "coordinate_scale": scale.tolist(),
                "input_dim": factory_kwargs["input_dim"]}, checkpoint_path)
    curves = pd.DataFrame(selected["curves"])
    curves.insert(0, "epoch_source", "inner")
    curves = pd.concat([curves, pd.DataFrame(outer_curves).assign(epoch_source="outer_refit")], ignore_index=True)
    curve_path = out / "curves" / f"curves_{run}.csv"
    curves.to_csv(curve_path, index=False)
    record = {"member": "M3", "exp_id": exp_id, "fold": fold_id,
              "seed": run_options.seeds[0], "best_epoch": best_epoch,
              "inner_best_evaal": selected["best_score"], "selected_loss_weights": selected_weights,
              "metrics": metrics, "inner_split": split_info,
              "candidate_inner_scores": [{"weights": x["weights"], "score": x["best_score"],
                                           "best_epoch": x["best_epoch"]} for x in candidate_runs],
              "config": {**factory_kwargs, **asdict(run_options)},
              "outer_train_users": sorted(int(x) for x in outer_train["USERID"].unique()),
              "outer_valid_users": sorted(int(x) for x in outer_valid["USERID"].unique()),
              "software": {"python": platform.python_version(), "torch": torch.__version__,
                           "sklearn": sklearn.__version__, "device": str(device)},
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "predictions_file": str(pred_path), "checkpoint_file": str(checkpoint_path),
              "curves_file": str(curve_path)}
    (out / "records" / f"record_{run}.json").write_text(json.dumps(record, indent=2, default=str))
    return record


def run_experiments(experiments: tuple[str, ...], folds: tuple[int, ...] | None = None,
                    seeds: tuple[int, ...] = (42, 1337, 2024),
                    run_options: M3RunOptions = M3RunOptions(),
                    skip_existing: bool = False, **kwargs) -> pd.DataFrame:
    train, _, _ = load_data()
    split_path = Path(__file__).resolve().parents[3] / "data/splits/folds.json"
    split = json.loads(split_path.read_text())
    if len(train) != split["train_rows"]:
        raise ValueError("Cleaned training rows no longer align with data/splits/folds.json")
    selected_folds = split["folds"] if folds is None else [f for f in split["folds"] if f["fold"] in folds]
    out = Path(kwargs.pop("output_dir", "results/phase2/m3"))
    records = []
    for exp_id in experiments:
        for fold in selected_folds:
            for seed in seeds:
                record_path = out / "records" / f"record_{exp_id}_f{fold['fold']}_s{seed}.json"
                if skip_existing and record_path.exists():
                    print(f"skip existing {record_path.name}", flush=True)
                    continue
                one_seed = M3RunOptions(**{**asdict(run_options), "seeds": (seed,)})
                record = run_m3_fold(exp_id, int(fold["fold"]), train,
                                     list(fold["train_idx"]), list(fold["valid_idx"]),
                                     run_options=one_seed, output_dir=out, **kwargs)
                records.append(record)
                print(f"saved {exp_id} fold={fold['fold']} seed={seed} "
                      f"best_epoch={record['best_epoch']} "
                      f"EvAAL={record['metrics']['evaal_style_score']:.3f}", flush=True)
    if skip_existing:
        wanted = set(experiments)
        records = [json.loads(path.read_text()) for path in sorted((out / "records").glob("record_*.json"))
                   if json.loads(path.read_text()).get("exp_id") in wanted]
    rows = []
    for record in records:
        rows.append({"exp_id": record["exp_id"], "fold": record["fold"],
                     "seed": record["seed"], "best_epoch": record["best_epoch"],
                     "inner_best_evaal": record["inner_best_evaal"], **record["metrics"]})
    result_frame = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    result_frame.to_csv(out / "m3_all_runs.csv", index=False)
    return result_frame
