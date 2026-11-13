"""Reproducible trainer for Phase 2 multitask indoor localization models."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.data.phase2 import (
    AugmentdBm,
    CoordinateScaler,
    WifiDataset,
    create_phase2_dataloaders,
    load_phase2_fold_data,
)
from src.evaluation.metrics import evaluate_predictions
from src.models.stub_multitask import build_multitask_model


@dataclass
class TrainConfig:
    """Experiment configuration for Phase 2 training runs."""

    exp_id: str
    fold: int
    seed: int
    dropout: float = 0.2
    batch_norm: bool = True
    weight_decay: float = 0.0
    lr: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 30
    patience: int = 8
    noise_sigma: float = 0.0
    ap_dropout: float = 0.0
    include_missing_mask: bool = False
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    # Loss weighting: building, floor, coordinate
    loss_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)


def compute_metrics_and_predictions(
    model: nn.Module,
    loader: DataLoader,
    coord_scaler: CoordinateScaler,
    device: torch.device,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Evaluate model on a DataLoader and compute canonical indoor localization metrics."""
    model.eval()
    pred_bldg_list, true_bldg_list = [], []
    pred_floor_list, true_floor_list = [], []
    pred_coords_list, true_coords_list = [], []
    user_ids, phone_ids, detected_counts, indices = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            feats = batch["features"].to(device)
            out = model(feats)

            pred_bldg = torch.argmax(out["building_logits"], dim=1).cpu().numpy()
            pred_floor = torch.argmax(out["floor_logits"], dim=1).cpu().numpy()
            pred_norm_coords = out["coordinates"].cpu().numpy()

            pred_bldg_list.append(pred_bldg)
            true_bldg_list.append(batch["building"].numpy())
            pred_floor_list.append(pred_floor)
            true_floor_list.append(batch["floor"].numpy())
            pred_coords_list.append(pred_norm_coords)
            true_coords_list.append(batch["coordinates"].numpy())

            user_ids.append(batch["userid"].numpy())
            phone_ids.append(batch["phoneid"].numpy())
            detected_counts.append(batch["detected_count"].numpy())
            indices.append(batch["index"].numpy())

    bldg_pred = np.concatenate(pred_bldg_list)
    bldg_true = np.concatenate(true_bldg_list)
    floor_pred = np.concatenate(pred_floor_list)
    floor_true = np.concatenate(true_floor_list)
    coords_pred_norm = np.concatenate(pred_coords_list)
    coords_true_norm = np.concatenate(true_coords_list)

    # Inverse transform coordinates back to true UTM metres
    coords_pred_meters = coord_scaler.inverse_transform(coords_pred_norm)
    coords_true_meters = coord_scaler.inverse_transform(coords_true_norm)

    delta_lon = coords_pred_meters[:, 0] - coords_true_meters[:, 0]
    delta_lat = coords_pred_meters[:, 1] - coords_true_meters[:, 1]
    pos_errors = np.hypot(delta_lon, delta_lat)

    preds_df = pd.DataFrame(
        {
            "index": np.concatenate(indices),
            "USERID": np.concatenate(user_ids),
            "PHONEID": np.concatenate(phone_ids),
            "detected_count": np.concatenate(detected_counts),
            "BUILDINGID_TRUE": bldg_true,
            "BUILDINGID_PRED": bldg_pred,
            "FLOOR_TRUE": floor_true,
            "FLOOR_PRED": floor_pred,
            "LONGITUDE_TRUE": coords_true_meters[:, 0],
            "LATITUDE_TRUE": coords_true_meters[:, 1],
            "LONGITUDE_PRED": coords_pred_meters[:, 0],
            "LATITUDE_PRED": coords_pred_meters[:, 1],
            "position_error": pos_errors,
        }
    )

    base_metrics = evaluate_predictions(preds_df)

    # Add Macro-F1 metrics
    base_metrics["building_macro_f1"] = float(
        f1_score(bldg_true, bldg_pred, average="macro", zero_division=0)
    )
    base_metrics["floor_macro_f1"] = float(
        f1_score(floor_true, floor_pred, average="macro", zero_division=0)
    )

    return base_metrics, preds_df


def compute_quick_evaal(
    model: nn.Module,
    loader: DataLoader,
    coord_scaler: CoordinateScaler,
    device: torch.device,
) -> float:
    """Fast tensor-level EvAAL score calculation for inner validation early stopping."""
    model.eval()
    bldg_ok_list, floor_ok_list, pos_err_list = [], [], []

    with torch.no_grad():
        for batch in loader:
            feats = batch["features"].to(device)
            out = model(feats)
            pred_bldg = torch.argmax(out["building_logits"], dim=1)
            pred_floor = torch.argmax(out["floor_logits"], dim=1)
            bldg_ok = pred_bldg == batch["building"].to(device)
            floor_ok = pred_floor == batch["floor"].to(device)

            norm_coords = out["coordinates"].cpu().numpy()
            true_norm = batch["coordinates"].numpy()
            pred_m = coord_scaler.inverse_transform(norm_coords)
            true_m = coord_scaler.inverse_transform(true_norm)
            errs = np.hypot(pred_m[:, 0] - true_m[:, 0], pred_m[:, 1] - true_m[:, 1])

            bldg_ok_list.append(bldg_ok.cpu().numpy())
            floor_ok_list.append(floor_ok.cpu().numpy())
            pos_err_list.append(errs)

    bldg_ok = np.concatenate(bldg_ok_list)
    floor_ok = np.concatenate(floor_ok_list)
    pos_err = np.concatenate(pos_err_list)
    evaal = np.mean(pos_err + 50.0 * (~bldg_ok) + 4.0 * (~floor_ok))
    return float(evaal)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_weights: tuple[float, float, float],
    device: torch.device,
) -> float:
    """Train model for a single epoch and return mean loss."""
    model.train()
    bldg_criterion = nn.CrossEntropyLoss()
    floor_criterion = nn.CrossEntropyLoss()
    coord_criterion = nn.MSELoss()
    w_b, w_f, w_c = loss_weights

    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        features = batch["features"].to(device)
        bldg_true = batch["building"].to(device)
        floor_true = batch["floor"].to(device)
        coords_true = batch["coordinates"].to(device)

        optimizer.zero_grad()
        out = model(features)

        l_b = bldg_criterion(out["building_logits"], bldg_true)
        l_f = floor_criterion(out["floor_logits"], floor_true)
        l_c = coord_criterion(out["coordinates"], coords_true)

        loss = w_b * l_b + w_f * l_f + w_c * l_c
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        num_batches += 1

    return total_loss / max(num_batches, 1)


def compute_eval_loss(
    model: nn.Module,
    loader: DataLoader,
    loss_weights: tuple[float, float, float],
    device: torch.device,
) -> float:
    """Compute average evaluation loss across DataLoader."""
    model.eval()
    bldg_criterion = nn.CrossEntropyLoss()
    floor_criterion = nn.CrossEntropyLoss()
    coord_criterion = nn.MSELoss()
    w_b, w_f, w_c = loss_weights

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            bldg_true = batch["building"].to(device)
            floor_true = batch["floor"].to(device)
            coords_true = batch["coordinates"].to(device)

            out = model(features)
            l_b = bldg_criterion(out["building_logits"], bldg_true)
            l_f = floor_criterion(out["floor_logits"], floor_true)
            l_c = coord_criterion(out["coordinates"], coords_true)

            loss = w_b * l_b + w_f * l_f + w_c * l_c
            total_loss += float(loss.item())
            num_batches += 1

    return total_loss / max(num_batches, 1)


def run_experiment_fold(
    config: TrainConfig,
    results_dir: Path | str = "results/phase2/m1",
) -> dict[str, Any]:
    """Execute a single fold run with inner grouped validation early stopping.
    
    Strict Out-Of-Fold Invariant:
    1. Outer validation fold is NEVER touched during inner training or early stopping.
    2. Outer training set is partitioned into inner train/val by USERID.
    3. Optimal stopping epoch E* is found by monitoring inner validation EvAAL score.
    4. Model is trained on full outer training users up to E* (or restored).
    5. Final evaluation is performed strictly once on outer validation fold.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)

    # Deterministic seeding
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # 1. Load fold data
    augmenter = (
        AugmentdBm(noise_sigma=config.noise_sigma, ap_dropout=config.ap_dropout)
        if (config.noise_sigma > 0 or config.ap_dropout > 0)
        else None
    )

    outer_train_ds, outer_val_ds, coord_scaler, meta = load_phase2_fold_data(
        fold_id=config.fold,
        augmenter=augmenter,
        include_missing_mask=config.include_missing_mask,
        seed=config.seed,
    )

    # 2. Inner Grouped Split within outer training users for early stopping
    gkf = GroupKFold(n_splits=4)
    inner_splits = list(gkf.split(outer_train_ds.raw_features, groups=outer_train_ds.user_ids))
    inner_train_idx, inner_val_idx = inner_splits[0]

    # Inner datasets
    inner_train_ds = Subset(outer_train_ds, inner_train_idx)
    # Validation dataset inside inner validation must NOT be augmented
    inner_val_raw_ds = WifiDataset(
        raw_features=outer_train_ds.raw_features[inner_val_idx],
        buildings=outer_train_ds.buildings[inner_val_idx],
        floors=outer_train_ds.floors[inner_val_idx],
        coordinates=outer_train_ds.coordinates[inner_val_idx],
        user_ids=outer_train_ds.user_ids[inner_val_idx],
        phone_ids=outer_train_ds.phone_ids[inner_val_idx],
        augmenter=None,  # No augmentation for validation
        include_missing_mask=config.include_missing_mask,
        seed=config.seed,
    )

    g = torch.Generator()
    g.manual_seed(config.seed)

    inner_train_loader = DataLoader(
        inner_train_ds, batch_size=config.batch_size, shuffle=True, generator=g
    )
    inner_val_loader = DataLoader(
        inner_val_raw_ds, batch_size=config.batch_size, shuffle=False
    )

    # 3. Inner model training to find best epoch E*
    input_dim = 1040 if config.include_missing_mask else 520
    inner_model = build_multitask_model(
        input_dim=input_dim,
        dropout=config.dropout,
        batch_norm=config.batch_norm,
        seed=config.seed,
    ).to(device)

    optimizer = torch.optim.Adam(
        inner_model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    best_inner_evaal = float("inf")
    best_epoch = 1
    patience_counter = 0

    for epoch in range(1, config.max_epochs + 1):
        train_epoch(inner_model, inner_train_loader, optimizer, config.loss_weights, device)
        inner_evaal = compute_quick_evaal(inner_model, inner_val_loader, coord_scaler, device)

        if inner_evaal < best_inner_evaal:
            best_inner_evaal = inner_evaal
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break

    # 4. Train final model on FULL outer training set for best_epoch epochs
    torch.manual_seed(config.seed)
    outer_model = build_multitask_model(
        input_dim=input_dim,
        dropout=config.dropout,
        batch_norm=config.batch_norm,
        seed=config.seed,
    ).to(device)

    outer_optimizer = torch.optim.Adam(
        outer_model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    full_train_loader = DataLoader(
        outer_train_ds, batch_size=config.batch_size, shuffle=True, generator=g
    )
    outer_val_loader = DataLoader(
        outer_val_ds, batch_size=config.batch_size, shuffle=False
    )

    epoch_curves: list[dict[str, float | int]] = []
    for epoch in range(1, best_epoch + 1):
        tr_loss = train_epoch(
            outer_model, full_train_loader, outer_optimizer, config.loss_weights, device
        )
        val_loss = compute_eval_loss(
            outer_model, outer_val_loader, config.loss_weights, device
        )
        epoch_curves.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": val_loss})

    # 5. Final outer validation evaluation
    final_metrics, final_preds_df = compute_metrics_and_predictions(
        outer_model, outer_val_loader, coord_scaler, device
    )

    # 6. Save predictions and run metadata
    run_name = f"{config.exp_id}_f{config.fold}_s{config.seed}"
    preds_file = results_dir / f"preds_{run_name}.csv"
    final_preds_df.to_csv(preds_file, index=False)

    run_record: dict[str, Any] = {
        "exp_id": config.exp_id,
        "fold": config.fold,
        "seed": config.seed,
        "best_epoch": best_epoch,
        "best_inner_evaal": float(best_inner_evaal),
        "final_train_loss": epoch_curves[-1]["train_loss"] if epoch_curves else float("nan"),
        "final_val_loss": epoch_curves[-1]["val_loss"] if epoch_curves else float("nan"),
        "metrics": final_metrics,
        "config": asdict(config),
        "software": {
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
            "python": platform.python_version(),
            "device": str(device),
        },
        "meta": meta,
        "epoch_curves": epoch_curves,
        "predictions_file": str(preds_file),
    }

    record_file = results_dir / f"record_{run_name}.json"
    with open(record_file, "w") as f:
        json.dump(run_record, f, indent=2)

    return run_record

