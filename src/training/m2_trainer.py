"""Member 2 Phase 2 trainer: representation (A4) and pretraining (A7) arms.

The epoch loop, loss, inner EvAAL stopping criterion and final metrics are
imported from Member 1's trainer, so an arm differs from the reference arm in
exactly one factor. Two deviations, both recorded in every run record: the
input view is a `RepresentationSpec`, and the inner split is
label-coverage-checked.

The out-of-fold invariant is unchanged - held-out users are read exactly once,
after E* is chosen inside outer training and the model refit on all
outer-training users.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import torch

from src.data.phase2 import AugmentdBm
from src.data.phase2_m2 import load_m2_fold_data, make_inner_split, make_loader
from src.features.rssi_representations import (
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    RepresentationSpec,
)
from src.models.dae import load_encoder_into_trunk
from src.models.stub_multitask import build_multitask_model
from src.training.phase2_trainer import (
    compute_eval_loss,
    compute_metrics_and_predictions,
    compute_quick_evaal,
    train_epoch,
)

DEFAULT_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
#: Trunk shape shared by the DAE encoder and the multitask model, so that the
#: A7 pretrained arm is architecture-matched to the random-init arm.
MATCHED_HIDDEN_DIMS = (256, 128)


@dataclass
class M2TrainConfig:
    """One Member 2 run: one arm, one outer fold, one seed."""

    exp_id: str
    fold: int
    seed: int
    representation: str = "linear"
    include_missing_mask: bool = False
    alpha: float = DEFAULT_ALPHA
    beta: float = DEFAULT_BETA
    hidden_dims: tuple[int, ...] = MATCHED_HIDDEN_DIMS
    dropout: float = 0.2
    batch_norm: bool = True
    weight_decay: float = 0.0
    lr: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 20
    patience: int = 6
    noise_sigma: float = 0.0
    ap_dropout: float = 0.0
    inner_split_mode: str = "stratified"
    inner_splits: int = 4
    pretrained_encoder: str | None = None
    device: str = DEFAULT_DEVICE
    loss_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def spec(self) -> RepresentationSpec:
        return RepresentationSpec(
            name=self.representation,
            include_missing_mask=self.include_missing_mask,
            alpha=self.alpha,
            beta=self.beta,
        )


def _build_model(config: M2TrainConfig, input_dim: int, device: torch.device):
    """Build the multitask model, optionally seeded from a DAE encoder."""
    model = build_multitask_model(
        input_dim=input_dim,
        dropout=config.dropout,
        batch_norm=config.batch_norm,
        seed=config.seed,
        hidden_dims=tuple(config.hidden_dims),
    )
    transfer_info: dict[str, Any] = {"pretrained": False}

    if config.pretrained_encoder:
        ckpt_path = Path(config.pretrained_encoder)
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if checkpoint.get("input_dim") != input_dim:
            raise ValueError(
                f"DAE checkpoint input_dim={checkpoint.get('input_dim')} does not match "
                f"model input_dim={input_dim}; refusing to transfer."
            )
        if checkpoint.get("fold") != config.fold:
            raise ValueError(
                f"DAE checkpoint was trained on fold {checkpoint.get('fold')} but this run is "
                f"fold {config.fold}; a pretrained encoder must be fold-matched."
            )
        transferred = load_encoder_into_trunk(
            model.trunk, checkpoint["encoder_state"], strict=True
        )
        transfer_info = {
            "pretrained": True,
            "checkpoint": str(ckpt_path),
            "checkpoint_fold": checkpoint.get("fold"),
            "checkpoint_representation": checkpoint.get("representation"),
            "transferred_tensors": len(transferred),
            "transferred_parameters": int(
                sum(int(np.prod(shape)) for shape in transferred.values())
            ),
        }

    return model.to(device), transfer_info


@dataclass
class FoldTraining:
    """Everything one fold's training produced, shared by A4/A7 and the A8 hybrid."""

    inner_model: torch.nn.Module
    outer_model: torch.nn.Module
    best_epoch: int
    best_inner_evaal: float
    inner_curve: list[dict[str, float | int]]
    epoch_curves: list[dict[str, float | int]]
    inner_train_ds: Any
    inner_val_ds: Any
    outer_train_ds: Any
    outer_val_ds: Any
    coord_scaler: Any
    inner_report: Any
    transfer_info: dict[str, Any]
    meta: dict[str, Any]
    input_dim: int
    device: torch.device


def train_fold_models(config: M2TrainConfig) -> FoldTraining:
    """Train the inner (stopping-epoch) and outer (refit) models for one fold.

    Step order is Member 1's: choose E* on an inner split of the outer-training
    users, then refit from scratch on all outer-training users for E* epochs.
    The held-out users are not touched here at all - only the caller evaluates
    them, once.
    """
    device = torch.device(config.device)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    spec = config.spec
    augmenter = (
        AugmentdBm(noise_sigma=config.noise_sigma, ap_dropout=config.ap_dropout)
        if (config.noise_sigma > 0 or config.ap_dropout > 0)
        else None
    )

    outer_train_ds, outer_val_ds, coord_scaler, meta = load_m2_fold_data(
        fold_id=config.fold, spec=spec, augmenter=augmenter, seed=config.seed
    )
    input_dim = spec.input_dim()

    inner_train_idx, inner_val_idx, inner_report = make_inner_split(
        user_ids=outer_train_ds.user_ids,
        buildings=outer_train_ds.buildings,
        floors=outer_train_ds.floors,
        n_splits=config.inner_splits,
        mode=config.inner_split_mode,
        seed=config.seed,
    )
    inner_train_ds = outer_train_ds.subset(inner_train_idx, augmenter=augmenter)
    inner_val_ds = outer_train_ds.subset(inner_val_idx, augmenter=None)

    inner_train_loader = make_loader(
        inner_train_ds, config.batch_size, True, config.seed
    )
    inner_val_loader = make_loader(inner_val_ds, config.batch_size, False, config.seed)

    inner_model, transfer_info = _build_model(config, input_dim, device)
    inner_optimizer = torch.optim.Adam(
        inner_model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    best_inner_evaal = float("inf")
    best_epoch = 1
    patience_counter = 0
    inner_curve: list[dict[str, float | int]] = []

    for epoch in range(1, config.max_epochs + 1):
        inner_train_loss = train_epoch(
            inner_model,
            inner_train_loader,
            inner_optimizer,
            config.loss_weights,
            device,
        )
        inner_evaal = compute_quick_evaal(
            inner_model, inner_val_loader, coord_scaler, device
        )
        inner_curve.append(
            {
                "epoch": epoch,
                "inner_train_loss": inner_train_loss,
                "inner_evaal": inner_evaal,
            }
        )
        if inner_evaal < best_inner_evaal:
            best_inner_evaal, best_epoch, patience_counter = inner_evaal, epoch, 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break

    torch.manual_seed(config.seed)
    outer_model, _ = _build_model(config, input_dim, device)
    outer_optimizer = torch.optim.Adam(
        outer_model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    full_train_loader = make_loader(
        outer_train_ds, config.batch_size, True, config.seed
    )
    outer_val_loader = make_loader(outer_val_ds, config.batch_size, False, config.seed)

    epoch_curves: list[dict[str, float | int]] = []
    for epoch in range(1, best_epoch + 1):
        tr_loss = train_epoch(
            outer_model, full_train_loader, outer_optimizer, config.loss_weights, device
        )
        val_loss = compute_eval_loss(
            outer_model, outer_val_loader, config.loss_weights, device
        )
        epoch_curves.append(
            {"epoch": epoch, "train_loss": tr_loss, "val_loss": val_loss}
        )

    return FoldTraining(
        inner_model=inner_model,
        outer_model=outer_model,
        best_epoch=best_epoch,
        best_inner_evaal=float(best_inner_evaal),
        inner_curve=inner_curve,
        epoch_curves=epoch_curves,
        inner_train_ds=inner_train_ds,
        inner_val_ds=inner_val_ds,
        outer_train_ds=outer_train_ds,
        outer_val_ds=outer_val_ds,
        coord_scaler=coord_scaler,
        inner_report=inner_report,
        transfer_info=transfer_info,
        meta=meta,
        input_dim=input_dim,
        device=device,
    )


def run_m2_fold(
    config: M2TrainConfig,
    results_dir: Path | str = "results/phase2/m2",
) -> dict[str, Any]:
    """Run one arm on one outer fold with one seed; save predictions and record."""
    results_dir = Path(results_dir)
    (results_dir / "preds").mkdir(parents=True, exist_ok=True)
    (results_dir / "records").mkdir(parents=True, exist_ok=True)

    trained = train_fold_models(config)
    outer_val_loader = make_loader(
        trained.outer_val_ds, config.batch_size, False, config.seed
    )

    # The held-out users are read exactly once, here.
    final_metrics, final_preds_df = compute_metrics_and_predictions(
        trained.outer_model, outer_val_loader, trained.coord_scaler, trained.device
    )

    run_name = f"{config.exp_id}_f{config.fold}_s{config.seed}"
    preds_file = results_dir / "preds" / f"preds_{run_name}.csv"
    final_preds_df.to_csv(preds_file, index=False)

    record: dict[str, Any] = {
        "member": "M2",
        "exp_id": config.exp_id,
        "fold": config.fold,
        "seed": config.seed,
        "representation": config.representation,
        "include_missing_mask": config.include_missing_mask,
        "input_dim": trained.input_dim,
        "best_epoch": trained.best_epoch,
        "best_inner_evaal": trained.best_inner_evaal,
        "final_train_loss": (
            trained.epoch_curves[-1]["train_loss"]
            if trained.epoch_curves
            else float("nan")
        ),
        "final_val_loss": (
            trained.epoch_curves[-1]["val_loss"]
            if trained.epoch_curves
            else float("nan")
        ),
        "metrics": final_metrics,
        "config": asdict(config),
        "pretraining": trained.transfer_info,
        "inner_split": asdict(trained.inner_report),
        "software": {
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "device": str(trained.device),
        },
        "meta": trained.meta,
        "inner_curve": trained.inner_curve,
        "epoch_curves": trained.epoch_curves,
        "predictions_file": str(preds_file),
    }

    with open(results_dir / "records" / f"record_{run_name}.json", "w") as f:
        json.dump(record, f, indent=2)

    return record
